import os
import argparse
import pandas as pd
import csv


def check_single_csv(path, required_cols=None):
    """
    단일 CSV 파일에 대해:
    - pandas로 읽히는지 확인
    - ParserError 발생 시 csv.reader로 몇 번째 줄에서 필드 수 깨졌는지 출력
    - (옵션) 필수 컬럼 누락 여부 검사
    """
    print(f"\n=== Checking: {path}")

    if not os.path.exists(path):
        print("  [SKIP] File does not exist.")
        return

    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError as e:
        print("  [PARSER ERROR]", e)

        # 어느 줄에서 필드 수가 깨지는지 추가 검사
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                expected = None
                for line_no, row in enumerate(reader, start=1):
                    fields = len(row)
                    if expected is None:
                        expected = fields
                    elif fields != expected:
                        print(
                            f"    [FIELD MISMATCH] line {line_no}: "
                            f"expected {expected} fields, got {fields} -> {row}"
                        )
        except Exception as e2:
            print("  [ERROR] While scanning with csv.reader:", e2)

        return True

    except Exception as e:
        print("  [OTHER ERROR]", e)
        return True

    print(f"  [OK] Parsed successfully. shape={df.shape}")
    print(f"  [OK] Columns: {list(df.columns)}")

    if required_cols is not None:
        missing = set(required_cols) - set(df.columns)
        if missing:
            print(f"  [WARN] Missing required columns: {missing}")
        else:
            print("  [OK] All required columns present.")
    
    return False

