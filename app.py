import logging

from aiohttp import web

from aqueduct.integrations.aiohttp import (
    FLOW_NAME,
    AppIntegrator,
)
from flow import (
    Flow,
    Task,
    get_flow
)


class EncoderView(web.View):
    @property
    def flow(self) -> Flow:
        return self.request.app[FLOW_NAME]

    async def post(self):
        im = await self.request.read()
        logging.warning(type(im))
        task = Task(im)
        await self.flow.process(task, timeout_sec=20)
        return web.json_response(data={'result': task.embedding})


def prepare_app() -> web.Application:
    app = web.Application(client_max_size=0)
    app.router.add_post('/get_embeddings', EncoderView)

    AppIntegrator(app).add_flow(get_flow())

    return app


if __name__ == '__main__':
    web.run_app(prepare_app())