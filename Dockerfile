# chewcall — containerized build (CPU / SIMD; the optional CUDA GPU backend is
# not included in this image). Multi-stage: build parasail + chewcall, then ship
# a slim runtime with the binaries and libparasail.so.
#
#   docker build -t chewcall:0.3.0 .
#   docker run --rm chewcall:0.3.0 --help
#   docker run --rm -v "$PWD":/data chewcall:0.3.0 \
#       -i /data/genomes -g /data/schema -o /data/out --cpu 8 --cds-input /data/cds

# ---- Stage 1: build ------------------------------------------------
FROM rust:1-bookworm AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build parasail (SIMD Smith-Waterman) -> /opt/parasail/build/libparasail.so
ARG PARASAIL_VERSION=v2.6.2
RUN git clone --depth 1 --branch ${PARASAIL_VERSION} \
        https://github.com/jeffdaily/parasail.git /opt/parasail \
    && cmake -S /opt/parasail -B /opt/parasail/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/parasail/build -j"$(nproc)"

# Build chewcall (and the auxiliary binaries) against parasail
WORKDIR /src
COPY . /src
ENV PARASAIL_DIR=/opt/parasail/build
RUN cargo build --release \
    && strip target/release/chewcall target/release/schema_audit \
             target/release/constructive_remedy target/release/schema_audit_pareto || true

# ---- Stage 2: runtime ----------------------------------------------
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/parasail/build/libparasail.so* /usr/local/lib/
COPY --from=build /src/target/release/chewcall            /usr/local/bin/
COPY --from=build /src/target/release/schema_audit        /usr/local/bin/
COPY --from=build /src/target/release/constructive_remedy /usr/local/bin/
COPY --from=build /src/target/release/schema_audit_pareto /usr/local/bin/
RUN ldconfig

ENV LD_LIBRARY_PATH=/usr/local/lib
WORKDIR /data
ENTRYPOINT ["chewcall"]
CMD ["--help"]
