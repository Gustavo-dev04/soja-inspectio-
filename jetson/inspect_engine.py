#!/usr/bin/env python3
"""Vígil.ia — mostra as entradas/saídas da engine TensorRT.

O app de inferência precisa saber NOME, FORMATO e TIPO de cada tensor pra
montar os buffers e interpretar a saída do RF-DETR (caixas + logits). Chutar
isso dá bug silencioso — melhor perguntar à engine.

Uso no Jetson:
    python3 inspect_engine.py soja_rfdetr_small_CAMPEAO_fp16.engine
"""
import sys

try:
    import tensorrt as trt
except ImportError:
    sys.exit('tensorrt não encontrado. No Jetson ele vem com o JetPack — '
             'tente: python3 -c "import tensorrt" com o python do sistema '
             '(não dentro de um venv sem --system-site-packages)')

path = sys.argv[1] if len(sys.argv) > 1 else 'soja_rfdetr_small_CAMPEAO_fp16.engine'

logger = trt.Logger(trt.Logger.WARNING)
with open(path, 'rb') as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())
if engine is None:
    sys.exit(f'não consegui carregar a engine: {path}\n'
             '(engine é atada ao aparelho + versão do TensorRT — reconstrua se '
             'o JetPack mudou)')

print(f'TensorRT : {trt.__version__}')
print(f'engine   : {path}')
print()

def linha(nome, entrada, shape, dtype):
    print(f'  {"ENTRADA" if entrada else "SAÍDA  "} | {nome:28s} | '
          f'shape={str(tuple(shape)):22s} | {dtype}')

if hasattr(engine, 'num_io_tensors'):          # TensorRT >= 10
    for i in range(engine.num_io_tensors):
        nome = engine.get_tensor_name(i)
        linha(nome,
              engine.get_tensor_mode(nome) == trt.TensorIOMode.INPUT,
              engine.get_tensor_shape(nome),
              engine.get_tensor_dtype(nome))
else:                                          # TensorRT 8.x
    for i in range(engine.num_bindings):
        linha(engine.get_binding_name(i),
              engine.binding_is_input(i),
              engine.get_binding_shape(i),
              engine.get_binding_dtype(i))

print()
print('O que importa pro app:')
print('  • shape da ENTRADA  -> pra que tamanho fazer o letterbox do frame')
print('  • shapes das SAÍDAS -> normalmente (1, N, 4) = caixas e (1, N, 5) ou')
print('    (1, N, 91) = logits por classe. N = nº de queries do DETR.')
print('  • se a última dimensão dos logits for > 5, o modelo guardou a cabeça')
print('    original do COCO e só as 5 primeiras colunas interessam.')
