import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from neo4j.exceptions import DriverError, Neo4jError

from app.api import chat, users
from app.db.neo4j import close_neo4j, init_neo4j
from app.db.sqlite import close_sqlite, init_sqlite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_neo4j()
    await init_sqlite()
    yield
    await close_sqlite()
    await close_neo4j()


app = FastAPI(title="Harness", lifespan=lifespan)
app.include_router(users.router)
app.include_router(chat.router)


@app.exception_handler(Neo4jError)
@app.exception_handler(DriverError)
async def graph_unavailable(request: Request, exc: Exception):
    logging.getLogger(__name__).exception("Graph database error")
    return JSONResponse(status_code=503, content={"detail": "User store temporarily unavailable"})


@app.get("/health")
async def health():
    return {"status": "ok"}
