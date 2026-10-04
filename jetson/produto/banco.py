"""Banco local do aparelho (SQLite): lotes, laudos e eventos.

Offline-first: tudo que o operador faz fica aqui, e a sincronização com a
nuvem (fase 2) só lê os laudos marcados como pendentes. Sem servidor, sem
dependência — `sqlite3` vem com o Python.

Migração por `PRAGMA user_version`: cada versão do esquema é um passo da lista
MIGRACOES, aplicado uma vez. Nunca editar um passo já publicado; acrescentar.
"""
import hashlib
import json
import sqlite3
import threading
import time
import uuid

MIGRACOES = [
    # 1 — esquema inicial
    """
    CREATE TABLE lotes (
        id            TEXT PRIMARY KEY,
        codigo        TEXT NOT NULL,
        produtor      TEXT DEFAULT '',
        variedade     TEXT DEFAULT '',
        amostra_kg    REAL,
        obs           TEXT DEFAULT '',
        estado        TEXT NOT NULL DEFAULT 'criado',
        criado_em     TEXT NOT NULL,
        iniciado_em   TEXT,
        encerrado_em  TEXT,
        parcial       TEXT
    );
    CREATE TABLE laudos (
        id              TEXT PRIMARY KEY,
        lote_id         TEXT NOT NULL REFERENCES lotes(id),
        aparelho_id     TEXT NOT NULL,
        modelo          TEXT NOT NULL,
        criado_em       TEXT NOT NULL,
        dados           TEXT NOT NULL,
        verificacao     TEXT NOT NULL,
        sync_estado     TEXT NOT NULL DEFAULT 'pendente',
        sync_tentativas INTEGER NOT NULL DEFAULT 0,
        sync_em         TEXT
    );
    CREATE INDEX laudos_sync ON laudos(sync_estado);
    CREATE TABLE eventos (
        id     INTEGER PRIMARY KEY AUTOINCREMENT,
        quando TEXT NOT NULL,
        tipo   TEXT NOT NULL,
        msg    TEXT NOT NULL
    );
    """,
]

ESTADOS_LOTE = ('criado', 'inspecionando', 'encerrado', 'interrompido')


def agora_iso():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def codigo_verificacao(dados):
    """SHA-256 do laudo em JSON canônico: qualquer alteração depois muda o
    código. É o que torna o laudo auditável — impresso, ele carrega a prova."""
    canon = json.dumps(dados, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(canon.encode()).hexdigest()


class Banco:
    def __init__(self, caminho):
        self.caminho = caminho
        self._trava = threading.Lock()
        self._con = sqlite3.connect(caminho, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute('PRAGMA journal_mode=WAL')   # leitura não trava escrita
        self._con.execute('PRAGMA foreign_keys=ON')
        self._migrar()

    def _migrar(self):
        with self._trava:
            v = self._con.execute('PRAGMA user_version').fetchone()[0]
            for i, sql in enumerate(MIGRACOES[v:], start=v + 1):
                self._con.executescript(sql)
                self._con.execute(f'PRAGMA user_version = {i}')
            self._con.commit()

    def versao(self):
        return self._con.execute('PRAGMA user_version').fetchone()[0]

    def _exec(self, sql, args=()):
        with self._trava:
            cur = self._con.execute(sql, args)
            self._con.commit()
            return cur

    def _um(self, sql, args=()):
        with self._trava:
            r = self._con.execute(sql, args).fetchone()
            return dict(r) if r else None

    def _todos(self, sql, args=()):
        with self._trava:
            return [dict(r) for r in self._con.execute(sql, args).fetchall()]

    # ---------------------------------------------------------------- lotes
    def criar_lote(self, codigo, produtor='', variedade='', amostra_kg=None, obs=''):
        codigo = str(codigo or '').strip()
        if not codigo:
            raise ValueError('o lote precisa de um código')
        if amostra_kg not in (None, ''):
            amostra_kg = float(amostra_kg)
            if amostra_kg <= 0:
                raise ValueError('amostra_kg tem que ser positiva')
        else:
            amostra_kg = None
        lid = str(uuid.uuid4())
        self._exec('INSERT INTO lotes (id, codigo, produtor, variedade, amostra_kg, obs, '
                   'criado_em) VALUES (?,?,?,?,?,?,?)',
                   (lid, codigo, str(produtor or ''), str(variedade or ''), amostra_kg,
                    str(obs or ''), agora_iso()))
        return self.lote(lid)

    def lote(self, lid):
        l = self._um('SELECT * FROM lotes WHERE id = ?', (lid,))
        if l and l.get('parcial'):
            l['parcial'] = json.loads(l['parcial'])
        return l

    def lotes(self, limite=50):
        return self._todos(
            'SELECT l.*, (SELECT id FROM laudos WHERE lote_id = l.id '
            'ORDER BY criado_em DESC LIMIT 1) AS laudo_id '
            'FROM lotes l ORDER BY criado_em DESC LIMIT ?', (limite,))

    def lote_ativo(self):
        return self._um("SELECT * FROM lotes WHERE estado = 'inspecionando' LIMIT 1")

    def marcar_lote(self, lid, estado):
        assert estado in ESTADOS_LOTE
        campo = {'inspecionando': 'iniciado_em', 'encerrado': 'encerrado_em',
                 'interrompido': 'encerrado_em'}.get(estado)
        if campo:
            self._exec(f'UPDATE lotes SET estado = ?, {campo} = ? WHERE id = ?',
                       (estado, agora_iso(), lid))
        else:
            self._exec('UPDATE lotes SET estado = ? WHERE id = ?', (estado, lid))

    def salvar_parcial(self, lid, dados):
        """Fotografia periódica do lote em andamento: se o aparelho desligar no
        meio, o lote não some — volta como 'interrompido' com estes números."""
        self._exec('UPDATE lotes SET parcial = ? WHERE id = ?',
                   (json.dumps(dados, ensure_ascii=False), lid))

    # ---------------------------------------------------------------- laudos
    def salvar_laudo(self, lote_id, aparelho_id, modelo, dados):
        lid = str(uuid.uuid4())
        criado = agora_iso()
        dados = dict(dados, laudo_id=lid, lote_id=lote_id, aparelho_id=aparelho_id,
                     modelo=modelo, emitido_em=criado)
        verif = codigo_verificacao(dados)
        self._exec('INSERT INTO laudos (id, lote_id, aparelho_id, modelo, criado_em, '
                   'dados, verificacao) VALUES (?,?,?,?,?,?,?)',
                   (lid, lote_id, aparelho_id, modelo, criado,
                    json.dumps(dados, ensure_ascii=False), verif))
        return self.laudo(lid)

    def laudo(self, lid):
        l = self._um('SELECT * FROM laudos WHERE id = ?', (lid,))
        if l:
            l['dados'] = json.loads(l['dados'])
            l['lote'] = self.lote(l['lote_id'])
            l['integro'] = codigo_verificacao(l['dados']) == l['verificacao']
        return l

    def laudos(self, pendentes=False, limite=50):
        sql = ('SELECT id, lote_id, aparelho_id, modelo, criado_em, verificacao, '
               'sync_estado FROM laudos')
        if pendentes:
            sql += " WHERE sync_estado = 'pendente'"
        return self._todos(sql + ' ORDER BY criado_em DESC LIMIT ?', (limite,))

    # ---------------------------------------------------------------- eventos
    def evento(self, tipo, msg):
        self._exec('INSERT INTO eventos (quando, tipo, msg) VALUES (?,?,?)',
                   (agora_iso(), tipo, str(msg)[:1000]))

    def eventos(self, limite=50):
        return self._todos('SELECT * FROM eventos ORDER BY id DESC LIMIT ?', (limite,))

    def fechar(self):
        with self._trava:
            self._con.close()
