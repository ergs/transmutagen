#
# Super simple example of a Dockerfile
#
FROM origen-base

WORKDIR /home/transmutagen

RUN cd /home/sympy && git pull

RUN cd /home/transmutagen && git pull

ENTRYPOINT ["python",  "-m", "transmutagen.origen"]
