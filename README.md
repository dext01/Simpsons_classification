# Simpsons Classification (Lab 1)
## Классификация персонажей мультсериала "Симпсоны" с использованием свёрточной нейронной сети ResNet18.
###  Описание задачи
```Цель: Построить модель для автоматической классификации изображений персонажей мультсериала "Симпсоны".
Датасет: The Simpsons Characters Dataset. (в датасете всего 20.933 фото, из низ 20 процентов на влаидацию и 80 на обучение, в некоторых папках с персонажем ~50 фото, в некоторых больше 1.000)

Архитектура: ResNet18 с заменой последнего слоя (full fine-tuning)

    Предобученные веса: ImageNet (IMAGENET1K_V1)
    Функция активации выхода: Softmax
    Функция потерь: CrossEntropyLoss
    Оптимизатор: Adam (lr=1e-4)
```

### Быстрый старт в google colab`e
#### 1. Клонируем форк репозитория и переходим в папку
```
!git clone https://github.com/dext01/Simpsons_classification - мой репозиторий
%cd Simpsons_classification
```
### 3. устанавливаем зависимости
```
!pip install -r requirements.txt
```
### 4. Импортируем датасет
```
import kagglehub

path = kagglehub.dataset_download("alexattia/the-simpsons-characters-dataset")

!rm -rf /content/Simpsons_classification/data

!ln -s {path}/simpsons_dataset/simpsons_dataset /content/Simpsons_classification/data
```
### 5. Как выглядит датасет?
```
    Simpsons_classification/
    └── data/
        ├── abraham_grampa_simpson/
        ├── agnes_skinner/
        ├── apu_nahasapeemapetilon/
        ├── ...
        └── (всего 42 папки с персонажами)
```

### 6. Запускаем тренировочный файл модели
```
python scripts/train.py --batch_size 32 --epochs 20 --lr 0.0001 --seed 42
```
#### 6.1 Что происходит на данном этапе?
```
1) Считываются входные параметры командной строки, по умолчанию или переданные:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    args = parser.parse_args()
```
```
2) Импортируются функции для подготовки датасета,
для подготовки модели ResNet18,
для прогона нашего датасета через модель.
```
```
3) Данные прогоняются через модель и вспомогательные методы, сохраняются веса в model.pth
```
```
Пример вывода в командную строку после данного этапа:

🚀 Using device: cuda
📊 Number of classes: 42
📦 Train batches: 524, Val batches: 131
Epoch 1/20 | Loss: 0.8173 | Val Acc: 92.52%
Epoch 2/20 | Loss: 0.1531 | Val Acc: 94.79%
Epoch 3/20 | Loss: 0.0423 | Val Acc: 94.72%
Epoch 4/20 | Loss: 0.0155 | Val Acc: 94.19%
Epoch 5/20 | Loss: 0.0153 | Val Acc: 93.65%
Epoch 6/20 | Loss: 0.0317 | Val Acc: 93.91%
Epoch 7/20 | Loss: 0.0320 | Val Acc: 93.60%
Epoch 8/20 | Loss: 0.0170 | Val Acc: 94.82%
Epoch 9/20 | Loss: 0.0131 | Val Acc: 94.24%
Epoch 10/20 | Loss: 0.0234 | Val Acc: 93.24%
Epoch 11/20 | Loss: 0.0137 | Val Acc: 94.86%
Epoch 12/20 | Loss: 0.0234 | Val Acc: 94.67%
Epoch 13/20 | Loss: 0.0158 | Val Acc: 94.65%
Epoch 14/20 | Loss: 0.0091 | Val Acc: 95.20%
Epoch 15/20 | Loss: 0.0126 | Val Acc: 94.22%
Epoch 16/20 | Loss: 0.0206 | Val Acc: 93.72%
Epoch 17/20 | Loss: 0.0103 | Val Acc: 95.25%
Epoch 18/20 | Loss: 0.0087 | Val Acc: 94.91%
Epoch 19/20 | Loss: 0.0138 | Val Acc: 93.98%
Epoch 20/20 | Loss: 0.0194 | Val Acc: 94.89%
```
```
4) Смотрим сохраненные артифакты
!ls -R artifacts/
(artifacts/:
model.pth  training_curve.png)

Выводим результат:
from IPython.display import Image, display
display(Image(filename='artifacts/training_curve.png'))
(images/1.png)
```
### 7. Запускаем валидацию
```
python scripts/val.py --model_path artifacts/model.pth --data_path ./data --save_cm artifacts/confusion_matrix.png
```
```
Пример моего выхода:


🚀 Using device: cuda
📊 Classes: 42
📦 Val batches: 131
✅ Model loaded: artifacts/model.pth
🔄 Running validation...

==================================================
📈 ОБЩИЕ МЕТРИКИ
==================================================
Total Samples: 4186
Average Loss:  0.0520
Accuracy:      99.09%
--------------------------------------------------
Macro Average (все классы равны):
  Precision:   0.9868
  Recall:      0.9846
  F1-Score:    0.9852
--------------------------------------------------
Weighted Average (учет размера класса):
  Precision:   0.9911
  Recall:      0.9909
  F1-Score:    0.9909
==================================================

📋 ПОДРОБНЫЙ ОТЧЕТ ПО КЛАССАМ:
                          precision    recall  f1-score   support

  abraham_grampa_simpson     0.9945    0.9945    0.9945       183
           agnes_skinner     0.8462    1.0000    0.9167        11
  apu_nahasapeemapetilon     1.0000    1.0000    1.0000       131
           barney_gumble     0.9583    0.9200    0.9388        25
            bart_simpson     0.9886    1.0000    0.9943       261
            carl_carlson     0.9412    1.0000    0.9697        16
charles_montgomery_burns     0.9837    0.9918    0.9877       243
            chief_wiggum     0.9801    0.9850    0.9825       200
         cletus_spuckler     1.0000    1.0000    1.0000         6
          comic_book_guy     0.9792    0.9691    0.9741        97
               disco_stu     1.0000    1.0000    1.0000         1
          edna_krabappel     1.0000    0.9783    0.9890        92
                fat_tony     1.0000    1.0000    1.0000         4
                     gil     1.0000    1.0000    1.0000         5
    groundskeeper_willie     0.9231    1.0000    0.9600        24
           homer_simpson     0.9954    0.9954    0.9954       432
           kent_brockman     1.0000    1.0000    1.0000       104
        krusty_the_clown     0.9920    0.9920    0.9920       251
           lenny_leonard     1.0000    0.9800    0.9899        50
             lionel_hutz     1.0000    1.0000    1.0000         1
            lisa_simpson     0.9889    0.9963    0.9926       269
          maggie_simpson     0.9600    0.9600    0.9600        25
           marge_simpson     0.9880    0.9960    0.9920       249
           martin_prince     1.0000    1.0000    1.0000        15
            mayor_quimby     1.0000    0.9636    0.9815        55
     milhouse_van_houten     0.9957    0.9871    0.9914       233
             miss_hoover     1.0000    1.0000    1.0000         4
             moe_szyslak     0.9932    0.9932    0.9932       293
            ned_flanders     0.9965    0.9894    0.9929       284
            nelson_muntz     0.9868    1.0000    0.9934        75
               otto_mann     1.0000    1.0000    1.0000         7
           patty_bouvier     1.0000    0.8824    0.9375        17
       principal_skinner     0.9958    0.9916    0.9937       237
    professor_john_frink     1.0000    1.0000    1.0000        10
      rainier_wolfcastle     1.0000    0.8889    0.9412         9
            ralph_wiggum     1.0000    0.9000    0.9474        20
           selma_bouvier     0.9565    1.0000    0.9778        22
            sideshow_bob     1.0000    1.0000    1.0000       167
            sideshow_mel     1.0000    1.0000    1.0000         8
          snake_jailbird     1.0000    1.0000    1.0000         8
            troy_mcclure     1.0000    1.0000    1.0000         1
         waylon_smithers     1.0000    1.0000    1.0000        41

                accuracy                         0.9909      4186
               macro avg     0.9868    0.9846    0.9852      4186
            weighted avg     0.9911    0.9909    0.9909      4186


🔥 Confusion Matrix (первые 10 классов для краткости):
[[182   0   0   0   0   0   0   0   0   0]
 [  0  11   0   0   0   0   0   0   0   0]
 [  0   0 131   0   0   0   0   0   0   0]
 [  0   0   0  23   0   0   0   0   0   1]
 [  0   0   0   0 261   0   0   0   0   0]
 [  0   0   0   0   0  16   0   0   0   0]
 [  0   0   0   0   1   0 241   0   0   0]
 [  0   0   0   0   0   0   1 197   0   0]
 [  0   0   0   0   0   0   0   0   6   0]
 [  0   0   0   1   0   0   0   1   0  94]]

✅ Confusion Matrix saved to: confusion_matrix.png

```
По аналогии с прошлым пунктом можем вывести матрицу ошибок
```
(images/confusion_matrix.png)
```
### Результат на тестовом датасете с кагла:
```
Accuracy: 98.79%
Macro Precision: 0.9002
Macro Recall: 0.8980
Macro F1: 0.8990
```
# Исползование оптимайзеров SGD и SGD+Momentum в обучении
```


✅ Random seed set to: 42
🚀 Using device: cuda
📊 Number of classes: 42

==================================================
Training with ADAM (lr=0.0001)
==================================================
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
100% 44.7M/44.7M [00:00<00:00, 136MB/s]
🏆 NEW BEST! Model saved at epoch 1 | Val Acc: 91.52%
Epoch 1/20 | Optimizer: adam | Loss: 0.8064 | Val Acc: 91.52% | Best: 91.52%
🏆 NEW BEST! Model saved at epoch 2 | Val Acc: 94.46%
Epoch 2/20 | Optimizer: adam | Loss: 0.1430 | Val Acc: 94.46% | Best: 94.46%
🏆 NEW BEST! Model saved at epoch 3 | Val Acc: 95.10%
Epoch 3/20 | Optimizer: adam | Loss: 0.0359 | Val Acc: 95.10% | Best: 95.10%
Epoch 4/20 | Optimizer: adam | Loss: 0.0149 | Val Acc: 94.84% | Best: 95.10%
Epoch 5/20 | Optimizer: adam | Loss: 0.0109 | Val Acc: 94.84% | Best: 95.10%
Epoch 6/20 | Optimizer: adam | Loss: 0.0239 | Val Acc: 93.22% | Best: 95.10%
Epoch 7/20 | Optimizer: adam | Loss: 0.0527 | Val Acc: 92.00% | Best: 95.10%
Epoch 8/20 | Optimizer: adam | Loss: 0.0213 | Val Acc: 94.55% | Best: 95.10%
Epoch 9/20 | Optimizer: adam | Loss: 0.0200 | Val Acc: 94.24% | Best: 95.10%
Epoch 10/20 | Optimizer: adam | Loss: 0.0134 | Val Acc: 93.62% | Best: 95.10%
Epoch 11/20 | Optimizer: adam | Loss: 0.0122 | Val Acc: 93.72% | Best: 95.10%
Epoch 12/20 | Optimizer: adam | Loss: 0.0241 | Val Acc: 94.27% | Best: 95.10%
Epoch 13/20 | Optimizer: adam | Loss: 0.0118 | Val Acc: 94.29% | Best: 95.10%
Epoch 14/20 | Optimizer: adam | Loss: 0.0183 | Val Acc: 94.53% | Best: 95.10%
Epoch 15/20 | Optimizer: adam | Loss: 0.0153 | Val Acc: 93.98% | Best: 95.10%
Epoch 16/20 | Optimizer: adam | Loss: 0.0091 | Val Acc: 94.39% | Best: 95.10%
Epoch 17/20 | Optimizer: adam | Loss: 0.0150 | Val Acc: 92.38% | Best: 95.10%
Epoch 18/20 | Optimizer: adam | Loss: 0.0149 | Val Acc: 94.79% | Best: 95.10%
Epoch 19/20 | Optimizer: adam | Loss: 0.0066 | Val Acc: 94.27% | Best: 95.10%
Epoch 20/20 | Optimizer: adam | Loss: 0.0113 | Val Acc: 93.81% | Best: 95.10%

✅ Final model saved: artifacts/optimizers_comparison/model_adam.pth
🏆 BEST model (epoch 3) saved: artifacts/optimizers_comparison/best_model_adam.pth | Val Acc: 95.10%
📊 Training curves saved to artifacts/optimizers_comparison/training_curve_adam.png
```
```
==================================================
Training with SGD (lr=0.01)
==================================================
🏆 NEW BEST! Model saved at epoch 1 | Val Acc: 87.27%
Epoch 1/20 | Optimizer: sgd | Loss: 1.0949 | Val Acc: 87.27% | Best: 87.27%
🏆 NEW BEST! Model saved at epoch 2 | Val Acc: 91.11%
Epoch 2/20 | Optimizer: sgd | Loss: 0.3338 | Val Acc: 91.11% | Best: 91.11%
🏆 NEW BEST! Model saved at epoch 3 | Val Acc: 93.14%
Epoch 3/20 | Optimizer: sgd | Loss: 0.1560 | Val Acc: 93.14% | Best: 93.14%
🏆 NEW BEST! Model saved at epoch 4 | Val Acc: 93.57%
Epoch 4/20 | Optimizer: sgd | Loss: 0.0796 | Val Acc: 93.57% | Best: 93.57%
🏆 NEW BEST! Model saved at epoch 5 | Val Acc: 93.84%
Epoch 5/20 | Optimizer: sgd | Loss: 0.0431 | Val Acc: 93.84% | Best: 93.84%
🏆 NEW BEST! Model saved at epoch 6 | Val Acc: 93.96%
Epoch 6/20 | Optimizer: sgd | Loss: 0.0277 | Val Acc: 93.96% | Best: 93.96%
🏆 NEW BEST! Model saved at epoch 7 | Val Acc: 94.39%
Epoch 7/20 | Optimizer: sgd | Loss: 0.0198 | Val Acc: 94.39% | Best: 94.39%
Epoch 8/20 | Optimizer: sgd | Loss: 0.0142 | Val Acc: 94.39% | Best: 94.39%
Epoch 9/20 | Optimizer: sgd | Loss: 0.0105 | Val Acc: 94.17% | Best: 94.39%
Epoch 10/20 | Optimizer: sgd | Loss: 0.0087 | Val Acc: 94.27% | Best: 94.39%
Epoch 11/20 | Optimizer: sgd | Loss: 0.0077 | Val Acc: 94.34% | Best: 94.39%
Epoch 12/20 | Optimizer: sgd | Loss: 0.0066 | Val Acc: 94.34% | Best: 94.39%
🏆 NEW BEST! Model saved at epoch 13 | Val Acc: 94.43%
Epoch 13/20 | Optimizer: sgd | Loss: 0.0058 | Val Acc: 94.43% | Best: 94.43%
🏆 NEW BEST! Model saved at epoch 14 | Val Acc: 94.53%
Epoch 14/20 | Optimizer: sgd | Loss: 0.0047 | Val Acc: 94.53% | Best: 94.53%
Epoch 15/20 | Optimizer: sgd | Loss: 0.0048 | Val Acc: 94.29% | Best: 94.53%
🏆 NEW BEST! Model saved at epoch 16 | Val Acc: 94.60%
Epoch 16/20 | Optimizer: sgd | Loss: 0.0042 | Val Acc: 94.60% | Best: 94.60%
Epoch 17/20 | Optimizer: sgd | Loss: 0.0038 | Val Acc: 94.51% | Best: 94.60%
Epoch 18/20 | Optimizer: sgd | Loss: 0.0037 | Val Acc: 94.58% | Best: 94.60%
Epoch 19/20 | Optimizer: sgd | Loss: 0.0034 | Val Acc: 94.58% | Best: 94.60%
Epoch 20/20 | Optimizer: sgd | Loss: 0.0030 | Val Acc: 94.39% | Best: 94.60%

✅ Final model saved: artifacts/optimizers_comparison/model_sgd.pth
🏆 BEST model (epoch 16) saved: artifacts/optimizers_comparison/best_model_sgd.pth | Val Acc: 94.60%
📊 Training curves saved to artifacts/optimizers_comparison/training_curve_sgd.png
```
```
==================================================
Training with SGD_MOMENTUM (lr=0.01)
==================================================
🏆 NEW BEST! Model saved at epoch 1 | Val Acc: 86.10%
Epoch 1/20 | Optimizer: sgd_momentum | Loss: 0.8647 | Val Acc: 86.10% | Best: 86.10%
🏆 NEW BEST! Model saved at epoch 2 | Val Acc: 90.35%
Epoch 2/20 | Optimizer: sgd_momentum | Loss: 0.2823 | Val Acc: 90.35% | Best: 90.35%
🏆 NEW BEST! Model saved at epoch 3 | Val Acc: 92.31%
Epoch 3/20 | Optimizer: sgd_momentum | Loss: 0.1266 | Val Acc: 92.31% | Best: 92.31%
🏆 NEW BEST! Model saved at epoch 4 | Val Acc: 93.57%
Epoch 4/20 | Optimizer: sgd_momentum | Loss: 0.0627 | Val Acc: 93.57% | Best: 93.57%
🏆 NEW BEST! Model saved at epoch 5 | Val Acc: 93.81%
Epoch 5/20 | Optimizer: sgd_momentum | Loss: 0.0409 | Val Acc: 93.81% | Best: 93.81%
Epoch 6/20 | Optimizer: sgd_momentum | Loss: 0.0409 | Val Acc: 93.14% | Best: 93.81%
🏆 NEW BEST! Model saved at epoch 7 | Val Acc: 94.10%
Epoch 7/20 | Optimizer: sgd_momentum | Loss: 0.0216 | Val Acc: 94.10% | Best: 94.10%
🏆 NEW BEST! Model saved at epoch 8 | Val Acc: 94.29%
Epoch 8/20 | Optimizer: sgd_momentum | Loss: 0.0213 | Val Acc: 94.29% | Best: 94.29%
🏆 NEW BEST! Model saved at epoch 9 | Val Acc: 94.79%
Epoch 9/20 | Optimizer: sgd_momentum | Loss: 0.0115 | Val Acc: 94.79% | Best: 94.79%
Epoch 10/20 | Optimizer: sgd_momentum | Loss: 0.0100 | Val Acc: 93.57% | Best: 94.79%
Epoch 11/20 | Optimizer: sgd_momentum | Loss: 0.0075 | Val Acc: 94.77% | Best: 94.79%
Epoch 12/20 | Optimizer: sgd_momentum | Loss: 0.0106 | Val Acc: 94.46% | Best: 94.79%
Epoch 13/20 | Optimizer: sgd_momentum | Loss: 0.0058 | Val Acc: 94.10% | Best: 94.79%
Epoch 14/20 | Optimizer: sgd_momentum | Loss: 0.0115 | Val Acc: 94.08% | Best: 94.79%
Epoch 15/20 | Optimizer: sgd_momentum | Loss: 0.0079 | Val Acc: 94.41% | Best: 94.79%
Epoch 16/20 | Optimizer: sgd_momentum | Loss: 0.0090 | Val Acc: 94.03% | Best: 94.79%
Epoch 17/20 | Optimizer: sgd_momentum | Loss: 0.0040 | Val Acc: 94.51% | Best: 94.79%
Epoch 18/20 | Optimizer: sgd_momentum | Loss: 0.0030 | Val Acc: 94.51% | Best: 94.79%
Epoch 19/20 | Optimizer: sgd_momentum | Loss: 0.0036 | Val Acc: 94.62% | Best: 94.79%
🏆 NEW BEST! Model saved at epoch 20 | Val Acc: 95.48%
Epoch 20/20 | Optimizer: sgd_momentum | Loss: 0.0011 | Val Acc: 95.48% | Best: 95.48%

✅ Final model saved: artifacts/optimizers_comparison/model_sgd_momentum.pth
🏆 BEST model (epoch 20) saved: artifacts/optimizers_comparison/best_model_sgd_momentum.pth | Val Acc: 95.48%
📊 Training curves saved to artifacts/optimizers_comparison/training_curve_sgd_momentum.png

============================================================
🏆 BEST RESULTS COMPARISON
============================================================
ADAM            | Best Val Acc: 95.10% | LR: 0.0001
SGD             | Best Val Acc: 94.60% | LR: 0.01
SGD_MOMENTUM    | Best Val Acc: 95.48% | LR: 0.01
============================================================

✅ All training completed! Results saved to artifacts/optimizers_comparison

```
## Выводы по сравнению оптимизаторов:

### 1. Скорость сходимости:
- Adam показал самую быструю сходимость - достиг 95% точности уже на 3-й эпохе
- SGD и SGD+momentum разгонялись медленнее, но стабильно наращивали точность

### 2. Стабильность обучения:
- Adam продемонстрировал нестабильную валидационную точность с колебаниями 
  (от 92% до 95%), что характерно для адаптивных методов из-за:
  • Индивидуального learning rate для каждого параметра
  • Склонности к осцилляциям вокруг минимума
  • Возможного попадания в острые минимумы функции потерь
  
- SGD и SGD+momentum показали плавную, монотонную сходимость благодаря:
  • Фиксированному размеру шага обучения
  • Эффекту сглаживания от momentum-термина
  • Тенденции к нахождению плоских минимумов, которые лучше обобщаются

### 3. Итоговая точность:
Все три оптимизатора достигли сопоставимой лучшей точности (~94-95%), 
что говорит о том, что для данной задачи с transfer learning (ResNet-18) 
выбор оптимизатора не критичен для финального результата.

### 4. Рекомендации:
- Для быстрой отладки и прототипирования: Adam (быстрая обратная связь)
- Для финального обучения и лучшего обобщения: SGD+momentum 
- Чистый SGD требует тщательного подбора learning rate, но даёт стабильный результат
