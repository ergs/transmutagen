#
# Super simple example of a Dockerfile
#
FROM origen-base

WORKDIR /

RUN cd /home/sympy && git pull

RUN cd /home/transmutagen && git pull

ENV PYTHONPATH=/home/sympy:/home/transmutagen

ENTRYPOINT ["python",  "-m", "transmutagen.origen_all"]
