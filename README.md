## Installation guide

```shell
pip install -r ./requirements.txt
```

запуск трейна

```
python train.py -c hw_hifi/configs/full_train_20000.json
```

После этого надо переложить из папки saved последний чекпоинт и его config в папку `default_test_model`, для запуска теста
переименовать их в `checkpoint.pth` и `config.json` соответственно

Можно использовать готовый checkpoint отсюда [https://drive.google.com/file/d/1iuJYUfrN17vOAXFEKZkCypZdffZh1zuy/view?usp=sharing](https://drive.google.com/file/d/1569yXrBtbxkfz-KhNCILjAVwvsXNwIgw/view?usp=sharing) вместе с full_train_20000.json из папки configs

запуск теста

```
python hw_hifi/test.py
```
Команда создаст сгенерированные аудио в папке `generated_audios`
