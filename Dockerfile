FROM ubuntu:22.04@sha256:3b06811b2afd352be909dd088a004166d665dc76d38b13eada33522a9d915c6f

ARG SNAPSHOT=https://snapshot.ubuntu.com/ubuntu/20260728T000000Z
ARG LIBC_VER=2.35-0ubuntu3.14
ARG GCC_VER=11.4.0-1ubuntu1~22.04.3
ARG CUDA_RUN=https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
ARG CUDA_MD5=a1ba6168710272c0f5eda622ca42172f
ARG CUDA_HOME=/usr/local/cuda-12.9
ARG NVCC_VER=12.9.41

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LC_ALL=C.UTF-8

RUN printf 'APT::Sandbox::User "root";\n' > /etc/apt/apt.conf.d/00-no-sandbox \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN printf 'deb %s jammy main universe\ndeb %s jammy-updates main universe\ndeb %s jammy-security main universe\n' \
        "$SNAPSHOT" "$SNAPSHOT" "$SNAPSHOT" > /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        libc6=$LIBC_VER \
        gcc-11=$GCC_VER \
        g++-11=$GCC_VER \
        make \
        python3 \
        wget \
        libxml2 \
        xz-utils \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
 && rm -rf /var/lib/apt/lists/*

RUN wget -q -O /tmp/cuda.run "$CUDA_RUN" \
 && printf '%s  /tmp/cuda.run\n' "$CUDA_MD5" | md5sum -c - \
 && TAR_OPTIONS=--no-same-owner sh /tmp/cuda.run --silent --toolkit --override \
 && rm -f /tmp/cuda.run \
 && rm -rf /var/log/cuda-installer.log /var/log/nvidia-installer.log

ENV PATH=${CUDA_HOME}/bin:${PATH} \
    CUDA_HOME=${CUDA_HOME}

RUN set -eu; \
    ok=1; \
    check() { \
        if [ "$2" = "$3" ]; then \
            echo "  PIN OK      $1 = $2"; \
        else \
            echo "  PIN FAILED  $1: got '$2', want '$3'" >&2; ok=0; \
        fi; \
    }; \
    got_nvcc="$(nvcc --version | sed -n 's/.*, V\([0-9][0-9.]*\).*/\1/p')"; \
    got_libc="$(dpkg-query -W -f='${Version}' libc6)"; \
    got_gxx="$(dpkg-query -W -f='${Version}' g++-11)"; \
    got_gxx_major="$(g++ -dumpversion | cut -d. -f1)"; \
    echo "--- environment pin assertions ---"; \
    check nvcc      "$got_nvcc"       "$NVCC_VER"; \
    check libc6     "$got_libc"       "$LIBC_VER"; \
    check g++-11    "$got_gxx"        "$GCC_VER"; \
    check "g++ (default alternative)" "$got_gxx_major" "11"; \
    [ "$ok" = 1 ] || { echo "environment does not match the measured configuration" >&2; exit 1; }; \
    echo "--- all pins match the GCP measured configuration ---"

COPY . /artifact
WORKDIR /artifact

RUN make \
 && test -x /artifact/main || { echo "make did not produce ./main" >&2; exit 1; }

RUN mkdir -p /artifact/out
ENV TMPDIR=/artifact/out

CMD ["./tests/run_all.sh"]
