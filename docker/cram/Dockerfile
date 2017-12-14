FROM ergs/ergs-base

WORKDIR /root

RUN apt-get update && apt-get install -y unzip libc6-i386 libc6 linux-headers-amd64 \
    && apt-get -y clean

RUN conda update conda && conda install -y numpy scipy sympy matplotlib pyne mpmath xonsh gcc scikit-umfpack jinja2 gmpy2 \
    && conda clean --all

RUN git clone https://github.com/ergs/transmutagen.git

RUN pip install iterm2_tools

ENV LD_LIBRARY_PATH=/opt/conda/lib

ENV MATPLOTLIBRC=/root/matplotlibrc
RUN mkdir -p $MATPLOTLIBRC
RUN echo "backend : Agg" > $MATPLOTLIBRC/matplotlibrc

WORKDIR /root/transmutagen

ENTRYPOINT ["docker/cram/pull_and_cram.sh"]
