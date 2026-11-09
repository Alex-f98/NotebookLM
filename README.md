## 1. Levantar Ollama (si no está corriendo)

```bash
ollama serve   # en otra terminal, si no quedó como servicio
```

## 2. Levantar la API

```bash
. .env_lm/bin/activate   # o desde la raíz con el venv
uvicorn app.main:app --reload
```

Deberías ver `Uvicorn running on http://127.0.0.1:8000`.

## 3. Usarla de dos formas

**Swagger UI** (la más fácil): abrí `http://localhost:8000/docs` en el navegador — ahí tenés botones para probar `/upload` y `/query` sin tocar curl.

**Con curl:**

```bash
# Subir el PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@reporte.pdf"

# Preguntar
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"¿Qué skills me faltan?"}'
```

