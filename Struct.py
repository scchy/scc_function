# python 3.6
# author: Scc_hy 
# create date: 2020-02-29
# Function： 数据结构


# ====================================================================
#                                   二叉树
#------ date: 2020-02-29
# ====================================================================
class BTree():
    """
    二叉树节点结构
    """
    def __init__(self, val = None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Create_BTree():
    """
    根据前项遍历(DLR)和左遍历(LDR)
    重构二叉树
    """
    def __init__(self, dlr, ldr):
        self.dlr = dlr
        self.ldr = ldr
        self._tree = self.reConstrauctBtree(self.dlr, self.ldr)

    def reConstrauctBtree(self, dlr, ldr):
        if dlr == []:
            return None
        ldr_dc = dict(zip(ldr, range(len(ldr))))
        n = ldr_dc[dlr[0]]
        root = Binary_Tree(dlr[0])
        print('node:', dlr[0], 'left :', dlr[1:n+1], 'right :', dlr[n+1:])
        root.left = reConstrauctBtree(dlr[1:n+1], ldr[:n])
        root.right = reConstrauctBtree(dlr[n+1:], ldr[n+1:])
        return root

    def pretree(self):
        """
        前项遍历
        """
        print(self._tree.val)
        if self._tree.left:
            self.pretree(self._tree.left)
        if root.right:
            self.pretree(self._tree.right)




# ====================================================================
#                                   链表
#------ date: 2020-02-29
# ====================================================================

class ChainNode():
    def __init__(self, val, pnext=None):
        self.val = val
        self._next = pnext


class createChain(ChainNode):
    """
    创建链表
    """
    def __init__(self, val_list, head=None):
        self.len = len(val_list)
        self.chain = ChainNode(head if head != None else None)
        self.create_chain(val_list)

    def create_chain(self, val_list: list) -> ChainNode:
        node_loop = self.chain
        for i in val_list:
            node_loop._next = ChainNode(i)
            node_loop = node_loop._next

    def _empt(self):
        """
        是否只有一个head
        """
        return self.len == 0

    def append(self, val):
        """ 
        在链表最后添加一个节点 或 链表
        """
        if isinstance(val, ChainNode):
            add_ = ChainNode(val)
            val_chain = val
            while val_chain:
                val_chain = val_chain._next
                self.len += 1
        else:
            add_ = val
            self.len += 1

        node_loop = self.chain
        while node_loop._next:
            node_loop = node_loop._next
        node_loop._next = add_
