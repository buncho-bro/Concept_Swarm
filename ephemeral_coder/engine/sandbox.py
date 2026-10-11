import ast
import traceback
import sys
import io

class SandboxRunner:
    """
    実行・検証サンドボックス (Sandbox Runner):
    1. 生成されたコードの ast.parse (構文チェック)
    2. 分離環境での動的実行検証 (exec)
    3. 失敗時の自動ポストモーテム圧縮
    """
    def __init__(self):
        pass

    def verify_and_test(self, code_str):
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
            exec(code_str, local_scope, local_scope)
            
            # process_data が存在する場合はテスト
            if "process_data" in local_scope:
                fn = local_scope["process_data"]
                try:
                    res1 = fn([-5, 0, 10, 20])
                    res2 = fn([]) # 空データ耐性
                except Exception as inner_e:
                    raise inner_e

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
        if "RecursionError" in err_type:
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
