from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI

from gateway.servises import xtts, wisper, ollama, comfy

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0, read=120.0))
    try:
        yield
    finally:
        await app.state.http.aclose()

app = FastAPI(title="Local AI Gateway", lifespan=lifespan)


from fastapi.openapi.utils import get_openapi
from pydantic.fields import FieldInfo
import typing as _t

def _walk(obj: _t.Any, path: _t.Tuple=_t.cast(_t.Tuple, ())):
    # генератор путей до всех вхождений FieldInfo (или чего-то экзотического)
    if isinstance(obj, FieldInfo):
        yield path, obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, path + (f"[{k!r}]",))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(v, path + (f"[{i}]",))
    else:
        return

def custom_openapi():
    # стандартный "сырой" dict перед сериализацией в pydantic OpenAPI-модель
    output = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
        description=app.description,
        terms_of_service=getattr(app, "terms_of_service", None),
        contact=getattr(app, "contact", None),
        license_info=getattr(app, "license_info", None),
        servers=getattr(app, "servers", None),
        tags=getattr(app, "openapi_tags", None),
        external_docs=getattr(app, "openapi_external_docs", None),
        webhooks=getattr(app, "webhooks", None),
        security_schemes=getattr(app, "security_schemes", None),
    )
    bad = list(_walk(output))
    if bad:
        print("=== OpenAPI contains FieldInfo (or unexpected objects) ===")
        for path, obj in bad:
            print("PATH:", " -> ".join(path))
            print("OBJECT:", repr(obj))
        # можно ещё грубо убрать, чтобы посмотреть, кто дальше упадёт
        def _strip(o):
            if isinstance(o, FieldInfo):
                return str(o)
            if isinstance(o, dict):
                return {k: _strip(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_strip(v) for v in o]
            return o
        output = _strip(output)

    return output

app.openapi = custom_openapi

app.include_router(xtts.router)
app.include_router(wisper.router)
app.include_router(ollama.router)
app.include_router(comfy.router)

@app.get("/health")
async def health():
    return {"ok": True}
