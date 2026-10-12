import ast
import traceback
import sys
import io

class SandboxRunner:
    """
    厳格受入検証サンドボックス (Strict Acceptance Sandbox):
    1. 構文解析 (AST Parse): 文法チェック
    2. 動的実行 (exec): 実行時エラーの検知
    3. 受入テスト（assert）の実行:
       単なるprint文や何もしないダミースタブ（printの罠）を検知し、AssertionError で即時リジェクト。
    4. 失敗時の自動ポストモーテム圧縮 & 斥力（忌避制約）導出
    """
    def __init__(self):
        pass

    def verify_and_test(self, code_str, acceptance_test_code=""):
        if not code_str or not code_str.strip():
            return {
                "success": False,
                "error_type": "EmptyCodeError",
                "root_cause": "生成されたコードが空です",
                "anti_pattern": "コードブロック内に有効なPythonコードを出力せよ。"
            }

        # 1. 構文解析 (AST Parse)
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return {
                "success": False,
                "error_type": "SyntaxError",
                "root_cause": f"構文エラー (Line {e.lineno}): {e.msg}",
                "anti_pattern": "不完全なインデントやコロンの欠落、閉じ括弧の不一致に注意せよ。"
            }

        # 2. 実行テスト (exec)
        local_scope = {}
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            # コード自体の定義を実行
            exec(code_str, local_scope, local_scope)

            # 3. 受入テスト（assert）が存在する場合は実行
            if acceptance_test_code and acceptance_test_code.strip():
                try:
                    exec(acceptance_test_code, local_scope, local_scope)
                except AssertionError as ae:
                    msg = str(ae) if str(ae) else "アサーション条件不一致"
                    return {
                        "success": False,
                        "error_type": "AssertionError",
                        "root_cause": f"受入アサーション失敗: {msg} (print逃げまたは未実装スタブを検知)",
                        "anti_pattern": "単なるprint出力やダミー戻り値で済ませず、受入テストのアサーション条件を満たす本物の計算ロジックを実装せよ。"
                    }

        except Exception as e:
            sys.stdout = old_stdout
            err_type = type(e).__name__
            err_msg = str(e)
            anti_pattern = self._derive_anti_pattern(err_type, err_msg)
            return {
                "success": False,
                "error_type": err_type,
                "root_cause": f"{err_type}: {err_msg}",
                "anti_pattern": anti_pattern
            }
        finally:
            sys.stdout = old_stdout

        return {
            "success": True,
            "stdout": redirected_output.getvalue()
        }

    def _derive_anti_pattern(self, err_type, err_msg):
        if "AssertionError" in err_type:
            return "受入アサーション条件を満たしていない。期待されるデータ構造と正確な計算ロジックを実装せよ。"
        elif "RecursionError" in err_type:
            return "再帰の基底条件が不十分で無限ループに陥った。再帰ではなくループ処理またはメモ化を採用せよ。"
        elif "ZeroDivisionError" in err_type:
            return "分母が0になるエッジケース（空データやゼロ値）のゼロ除算ガードが欠落している。"
        elif "KeyError" in err_type or "IndexError" in err_type:
            return "存在しないキーまたはインデックスに無防備にアクセスした。辞書の.get()や境界値チェックを行え。"
        elif "TypeError" in err_type:
            return f"型の不一致が発生している（{err_msg}）。入力値の型検証と安全なキャストを徹底せよ。"
        elif "NameError" in err_type:
            return f"未定義の変数または未インポートのモジュールを参照した（{err_msg}）。"
        else:
            return f"実行時例外（{err_type}）が発生。防御的プログラミングとエッジケースのバリデーションを強化せよ。"
