
# Risk/Reward Filter

**Risk/Reward Filter** — это Python-модуль для фильтрации торговых сигналов по правилу *Risk/Reward ≥ 2:1*.  
Подходит для интеграции в любые торговые системы и бэктесты.

## 📌 Возможности
- Расчёт R:R и целевых цен.
- Фильтрация сделок по минимальному R:R.
- Автоматический расчёт размера позиции по фиксированному риску (% от депозита).
- Поддержка ATR-стопов.
- Векторизованная обработка DataFrame (Pandas).

## 🚀 Установка
```bash
# Клонируем репозиторий
git clone https://github.com/trade-stasvinokur/rr-filter-trading-steenbarger.git
cd rr-filter-trading-steenbarger/src
uv sync
```

## 🛠 Пример использования
```python
import pandas as pd
from rr_filter_template import filter_trades_by_rr

# Пример DataFrame со сделками
df = pd.DataFrame({
    "symbol": ["AAPL", "MSFT", "TSLA", "NVDA"],
    "direction": ["long", "short", "long", "short"],
    "entry": [200.0, 420.0, 250.0, 120.0],
    "stop":  [195.0, 430.0, 245.0, 130.0],
    "target":[212.0, 394.0, 262.0, 108.0]
})

# Фильтрация по правилу R:R ≥ 2.0
filtered = filter_trades_by_rr(df, min_rr=2.0)
print(filtered)
```

## 📊 Пример результата
| symbol | direction | entry  | stop   | target | rr   | pass_rr |
|--------|-----------|--------|--------|--------|------|---------|
| AAPL   | long      | 200.0  | 195.0  | 212.0  | 2.40 | True    |
| MSFT   | short     | 420.0  | 430.0  | 394.0  | 2.60 | True    |
| TSLA   | long      | 250.0  | 245.0  | 262.0  | 2.40 | True    |
| NVDA   | short     | 120.0  | 130.0  | 108.0  | 1.20 | False   |

## 📄 Лицензия
MIT License
