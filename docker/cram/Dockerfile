FROM ergs/ergs-base

WORKDIR /home

RUN apt-get update && apt-get install -y unzip libc6-i386 libc6 linux-headers-amd64
RUN apt-get -y clean

RUN conda install -y numpy scipy sympy matplotlib pyne mpmath xonsh gcc scikit-umfpack jinja2 gmpy2
RUN conda remove -y sympy
RUN conda clean --all

RUN git clone https://github.com/ergs/transmutagen.git

WORKDIR /home/transmutagen

ENTRYPOINT ["docker/cram/pull_and_cram.sh"]
