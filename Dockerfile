FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir --upgrade -r /app/requirements.txt

# RUN pip install --no-cache-dir --upgrade pip setuptools && \ pip install --no-cache-dir --upgrade -r /app/requirements.txt

# COPY ./app /app/app && \
#     ./app/model/data/recipe/data/all-recipe-cleaned.xlsx /path/in/container/ && \
#     ./app/model/data/recipe/data/small_ratings.xlsx.xlsx /path/in/container/ && \
#     ./app/model/data/recipe/data/harga-bahan-cleaned.xlsx /path/in/container/ && \
#     ./app/model/data/user_ratings/small-W-final.xlsx /path/in/container/ && \
#     ./app/model/data/user_ratings/small-X-final.xlsx /path/in/container/ && \
#     ./app/model/data/user_ratings/small-B-final.xlsx /path/in/container/ && \
#     ./app/model/RecomendationV2.h5 /path/in/container/ 

COPY ./app /app/app
COPY ./app/model/data/recipe/all-recipe-cleaned.xlsx /path/in/container/
COPY ./app/model/data/recipe/small_ratings.xlsx /path/in/container/
COPY ./app/model/data/recipe/harga-bahan-cleaned.xlsx /path/in/container/
COPY ./app/model/data/user_ratings/small-W-final.xlsx /path/in/container/
COPY ./app/model/data/user_ratings/small-X-final.xlsx /path/in/container/
COPY ./app/model/data/user_ratings/small-B-final.xlsx /path/in/container/
COPY ./app/model/RecomendationV2.h5 /path/in/container/



