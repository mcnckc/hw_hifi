## Installation guide

```shell
pip install -r ./requirements.txt
```

запуск трейна

```
python hw_asr/train.py -c hw_asr/hw_asr/configs/final.json
```

После этого надо переложить из папки saved последний чекпоинт и его config в папку `default_test_model`, для запуска теста
переименовать их в `checkpoint.pth` и `config.json` соответственно
Можно использовать готовый checkpoint отсюда https://drive.google.com/file/d/1iuJYUfrN17vOAXFEKZkCypZdffZh1zuy/view?usp=sharing вместе с final.json из папки configs

запуск теста

```
python hw_asr/test.py
```
