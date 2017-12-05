FROM origen-base

WORKDIR /

RUN cd /home/transmutagen && git pull

ENV PYTHONPATH=/home/transmutagen:/home/transmutagen/py_solve

ENTRYPOINT ["python",  "-m", "transmutagen.origen_all"]
