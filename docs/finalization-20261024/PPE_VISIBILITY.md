# Видимость СИЗ: диагностические реальные кадры

17 вручную просмотренных примеров, 3 лицензированных видео Intel. Разметка выполнена агентом по изображению, без второго независимого разметчика; границы приблизительные. Это диагностическая выборка, не оценка качества внедрения.

Исходные GT-рамки используются только после инференса для оценки. В модель передаются настоящие кадры и ROI от настоящего person detector. Ни один инцидент по этой таблице не создаётся. CPU PPE ONNX, не LinQ. FP/FN ниже относятся к классификации отсутствия СИЗ, а не к нарушениям правил в офисе или на улице.

| Кадр | Человек px / модель px | Голова px / корпус px | GT каска/жилет | Full-frame | ROI-auto |
|---|---|---|---|---|---|
| one-by-one-person-detection-8-person | 376.8 / 311.8 | [70.80000000000001, 79.2] / [120.0, 195.60000000000002] | missing/missing | unknown/unknown | unknown/unknown |
| one-by-one-person-detection-10-person | 380.4 / 313.1 | [67.19999999999999, 82.8] / [114.0, 199.2] | missing/missing | unknown/unknown | unknown/unknown |
| one-by-one-person-detection-40-person | 382.8 / 312.3 | [58.799999999999955, 72.0] / [108.0, 145.2] | missing/missing | unknown/unknown | unknown/unknown |
| one-by-one-person-detection-60-person | 396.0 / 324.9 | [82.79999999999995, 96.0] / [122.39999999999998, 196.8] | unknown/missing | unknown/unknown | unknown/unknown |
| one-by-one-person-detection-80-person | 356.4 / 288.9 | [56.400000000000034, 76.8] / [111.59999999999997, 144.0] | missing/missing | unknown/unknown | unknown/unknown |
| one-by-one-person-detection-100-person | 367.2 / 308.7 | [74.40000000000003, 91.2] / [122.39999999999998, 174.0] | missing/missing | unknown/unknown | unknown/unknown |
| person-bicycle-car-detection-3-person | 104.4 / 86.1 | [24.0, 25.19999999999999] / [51.60000000000002, 39.599999999999966] | unknown/missing | unknown/unknown | unknown/unknown |
| person-bicycle-car-detection-4-person | 102.0 / 80.9 | [24.0, 21.599999999999994] / [48.0, 39.599999999999966] | unknown/missing | unknown/unknown | unknown/unknown |
| person-bicycle-car-detection-5-person | 92.4 / 77.2 | [21.600000000000023, 22.80000000000001] / [42.0, 33.599999999999994] | unknown/missing | unknown/unknown | unknown/unknown |
| person-bicycle-car-detection-6-person | 97.2 / 74.0 | [22.800000000000068, 22.80000000000001] / [43.200000000000045, 38.400000000000006] | unknown/missing | unknown/unknown | unknown/unknown |
| worker-zone-detection-6-person | 666.0 / 220.2 | [93.0, 114.0] / [147.0, 240.0] | present/present | present/unknown | present/unknown |
| worker-zone-detection-25-person | 630.0 / 211.5 | [93.0, 123.0] / [195.0, 225.0] | present/present | present/unknown | present/unknown |
| worker-zone-detection-40-left | 510.0 / 169.6 | [87.0, 105.0] / [126.0, 186.0] | present/present | unknown/unknown | present/unknown |
| worker-zone-detection-40-right | 483.0 / 154.1 | [84.0, 96.0] / [117.0, 210.0] | present/present | unknown/unknown | present/unknown |
| worker-zone-detection-55-person | 501.0 / 163.1 | [90.0, 93.0] / [111.0, 201.0] | present/present | present/unknown | unknown/unknown |
| worker-zone-detection-65-person | 654.0 / 206.5 | [108.0, 132.0] / [195.0, 294.0] | present/present | present/unknown | present/unknown |
| worker-zone-detection-72-person | 555.0 / 182.5 | [96.0, 114.0] / [168.0, 267.0] | present/present | present/unknown | present/unknown |

При текущих порогах: в обоих режимах **5 FN отсутствия каски и 10 FN отсутствия жилета**, 0 FP этих классов. UNKNOWN не засчитывается как успешное распознавание. Из 7 случаев видимой каски full-frame распознал 5, ROI — 6. Жилет остался UNKNOWN во всех 17 примерах. Слабые ответы сохранены в ppe-detections.json; пороги не изменялись.

Во всех этих примерах размер обнаруженного человека выше 72 px: выборка не калибрует нижнюю границу. Пропуски при крупном человеке показывают, что размер сам по себе не гарантирует успех. Причина ошибки модели неизвестна; яркость/резкость и размеры — измерения, не доказательство причинности. Порог 72 остаётся консервативной некалиброванной политикой допуска, не утверждением невозможности обнаружения.

Полные размеры, яркость/резкость, FP/FN каждой строки, точные timestamps и причины UNKNOWN: [CSV](ppe-visibility.csv) и [JSON](ppe-visibility.json). Исходные прямоугольники для повторной проверки: [разметка](ppe-visibility-labels.json).
