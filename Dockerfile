# InterviewGenie — container image.
#
# The application is pure standard-library Python 3.11, so the image is tiny and
# needs no build step.  Everything that varies at runtime is an environment
# variable (see .env.example), which keeps the image the same in dev and prod.

FROM python:3.11-slim

# InterviewGenie has no third-party dependencies: no pip install, no wheels,
# no build toolchain.  That is deliberate — it makes the image small and
# removes an entire class of supply-chain and CVE work.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INTERVIEWGENIE_LOG_LEVEL=INFO \
    INTERVIEWGENIE_HOST=0.0.0.0 \
    INTERVIEWGENIE_PORT=8420

WORKDIR /app

COPY interviewgenie ./interviewgenie
COPY config ./config
COPY web ./web

# The knowledge-graph snapshot under data/ is generated on first start, so it is
# not copied in; the app recreates it if it is missing.
RUN mkdir -p data && chown genie:genie data

# Never run as root.
RUN useradd --create-home --shell /usr/sbin/nologin genie \
    && chown -R genie:genie /app
USER genie

EXPOSE 8420

# The container has no GPU and no audio device, so system-audio capture has to
# happen in the browser; the server only ever sees text and answer streams.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8420/api/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "interviewgenie", "serve", \
     "--host", "0.0.0.0", "--port", "8420"]
