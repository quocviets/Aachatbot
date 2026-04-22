docker run --name pgvector-container \
  -e POSTGRES_PASSWORD=111789 \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16