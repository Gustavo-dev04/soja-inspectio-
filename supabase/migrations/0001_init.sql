-- Schema inicial do soja-inspection.
-- Aplicar no projeto Supabase (SQL Editor) ou via `apply_migration`.

create table if not exists inspecoes (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz default now(),
  imagem_url text,
  total_graos integer,
  resultado_json jsonb
);

create table if not exists lotes (
  id uuid primary key default gen_random_uuid(),
  nome text not null,
  data date default current_date,
  total_inspecoes integer default 0,
  percentual_defeito numeric(5,2) default 0
);

alter table inspecoes enable row level security;
alter table lotes enable row level security;

-- O backend usa a service_role (bypassa RLS). O frontend usa a anon key:
-- precisa de leitura pública em ambas e insert público em inspecoes.
drop policy if exists "public read inspecoes" on inspecoes;
create policy "public read inspecoes" on inspecoes for select using (true);

drop policy if exists "public insert inspecoes" on inspecoes;
create policy "public insert inspecoes" on inspecoes for insert with check (true);

drop policy if exists "public read lotes" on lotes;
create policy "public read lotes" on lotes for select using (true);

drop policy if exists "public insert lotes" on lotes;
create policy "public insert lotes" on lotes for insert with check (true);
