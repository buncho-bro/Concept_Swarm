import json
import uuid

class GoalNode:
    """
    目標樹形図の構成ノード (Goal Tree Node):
    単なるタスク名ではなく、アトミック性判定、受入テスト契約、実行状態を保持する。
    """
    def __init__(self, node_id, name, parent_id=None, depth=0):
        self.id = node_id
        self.name = name
        self.parent_id = parent_id
        self.depth = depth
        self.children = []
        self.is_atomic = False
        self.status = "UNEXPANDED"  # UNEXPANDED, EXPANDING, BRANCH, ATOMIC, EXECUTING, VERIFIED, FAILED, PRUNED
        self.acceptance_test = ""
        self.code = ""
        self.worker_id = None
        self.error_log = None

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "children": self.children,
            "is_atomic": self.is_atomic,
            "status": self.status,
            "acceptance_test": self.acceptance_test,
            "worker_id": self.worker_id,
            "error_log": self.error_log
        }

class GoalTree:
    """
    目標樹形図 (Goal Tree):
    Root（完成形の要求）から逆算し、
    すべての末端ノードが null（不可分なアトミック単位）に到達するまで管理する。
    """
    def __init__(self, root_goal):
        self.tree_id = f"tree_{str(uuid.uuid4())[:6]}"
        self.root_id = "node_0"
        self.nodes = {}
        self.counter = 1
        self.acceptance_suite = ""
        self.decomposition_steps = []

        # Root ノード登録
        root_node = GoalNode(self.root_id, root_goal, parent_id=None, depth=0)
        self.nodes[self.root_id] = root_node

    def get_deepest_unexpanded_node(self):
        """
        『現在最も遠い所にある要素』を特定する。
        未展開ノードの中で深度（depth）が最大のものを優先し、同一深度なら最古のものを返す。
        """
        unexpanded = [n for n in self.nodes.values() if not n.is_atomic and n.status == "UNEXPANDED"]
        if not unexpanded:
            return None
        # depth 降順でソート
        unexpanded.sort(key=lambda x: x.depth, reverse=True)
        return unexpanded[0]

    def add_child(self, parent_node, child_name):
        child_id = f"node_{self.counter}"
        self.counter += 1
        child_node = GoalNode(child_id, child_name, parent_id=parent_node.id, depth=parent_node.depth + 1)
        self.nodes[child_id] = child_node
        parent_node.children.append(child_id)
        return child_node

    def is_complete(self):
        """
        すべての最終要素が null（不可分なアトミック単位）に到達したか判定
        """
        unexpanded = [n for n in self.nodes.values() if n.status == "UNEXPANDED"]
        return len(unexpanded) == 0

    def get_atomic_leaves(self):
        """
        実装可能なアトミック葉ノード群をボトムアップ順（最深ノードから）で取得
        """
        leaves = [n for n in self.nodes.values() if n.is_atomic and len(n.children) == 0]
        leaves.sort(key=lambda x: x.depth, reverse=True)
        return leaves

    def to_dict(self):
        return {
            "tree_id": self.tree_id,
            "root_id": self.root_id,
            "root_goal": self.nodes[self.root_id].name,
            "total_nodes": len(self.nodes),
            "is_complete": self.is_complete(),
            "acceptance_suite": self.acceptance_suite,
            "decomposition_steps": self.decomposition_steps,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()}
        }
