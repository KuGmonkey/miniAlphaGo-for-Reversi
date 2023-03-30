import copy
import math
import random
#节点类
class Node:
    def __init__(self, state, parent=None, action=None, color="X"):
        self.color = color #该节点玩家颜色（黑棋或者白棋）
        self.parent = parent #父节点
        self.children = [] #子节点
        self.reward = 0.0 #总得分
        self.state = state #棋盘状态
        self.visits = 0  #访问次数
        self.action = action #从父节点转移到子节点采取的动作

    # 增加子节点，每从可以落子的位置中挑选一个扩展，就增加一个子节点
    def add_child(self, child_state, action, color):
        child_node = Node(child_state, parent=self, action=action, color=color)
        self.children.append(child_node)

    # 判断是否还有未扩展的节点
    def fully_expanded(self):
        action = list(self.state.get_legal_actions(self.color))
        #当子节点数与合法落子位置数一致时，证明该节点全部扩展
        if len(self.children) == len(action):
            return True
        return False
class AIPlayer:
    """
    AI 玩家
    """
    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        #玩家颜色
        self.color = color
        
        # 最大迭代次数，即模拟下棋的次数，开辟尽可能多的路径，让AI学习尽可能多的不同的走法，
        # 得到尽可能多的不同的终局得分，便于ucb出最优得分，但也要考虑时间限制
        self.max_times = 250

        # UCB超参数，越大，越倾向于探索
        self.SCALAR = 2

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------

        #加载当前棋盘
        board_state = copy.deepcopy(board)
        #加载当前根节点
        root = Node(state=board_state, color=self.color)
        #蒙特卡洛搜索，搜索出最优的得分（经过max_times次的模拟探索，AI已经知道了大部分不同走法的得分，只需根据ucb值选择即可）
        action = self.Search(self.max_times, root)
        # ------------------------------------------------------------------------
        return action
    
    #蒙特卡洛树搜索具体实现，根据当前盘面获取最佳落子位置
    def Search(self, max_times, root):
        #重复搜索max_time次，扩展尽可能多的节点
        for t in range(max_times):
            #选择扩展
            expand_node = self.Select(root)
            #模拟
            reward = self.Stimulate(expand_node)
            #回溯
            self.Back(expand_node, reward)
            #选择当前盘面的最佳落子位置
            best_child = self.ucb(root, 0)
        return best_child.action

    #选择并扩展节点，返回被扩展的节点
    def Select(self, node):
        #游戏没结束则继续
        while not self.is_end(node.state):
            #如果当前根节点一个都没扩展过，那么直接扩展一个子节点并返回
            if len(node.children) == 0:
                return self.expand(node)
            #扩展过，但没扩展干净，则0.8的概率再扩展（expand）一个并返回，0.2的概率在已扩展的节点里选一个ucb值最大的继续选择过程
            else:
                if not node.fully_expanded():
                    if random.uniform(0, 1) < 0.8:
                        return self.expand(node)
                    else:
                        node=self.ucb(node, self.SCALAR)
                #扩展干净了，则选子节点中ucb最大的继续选择过程
                else:
                    node=self.ucb(node, self.SCALAR)
        return node

    #扩展的实现
    def expand(self, node):
        # 列出所有合法动作
        action_list = list(node.state.get_legal_actions(node.color))
        # 没有合法落子点，则返回父节点，进入模拟阶段
        if len(action_list) == 0:
            return node.parent

        #在合法落子点中随机选一个，若这个点已经被扩展过了（即node.child），则重新在合法落子位置选一个，再判断，直到找到未扩展过的节点
        action = random.choice(action_list)
        tried_action = [c.action for c in node.children]
        while action in tried_action:
            action = random.choice(action_list)

        # 复制状态并根据扩展的节点更新到新状态
        new_state = copy.deepcopy(node.state)
        new_state._move(action, node.color)

        # 确定子节点颜色
        if node.color == 'X':
            new_color = 'O'
        else:
            new_color = 'X'

        # 新建子节点
        node.add_child(new_state, action=action, color=new_color)
        return node.children[-1]

    #循环遍历子节点，若未被访问过，直接返回该节点，否则在得分最高的子节点中，随机选一个
    def ucb(self, node, scalar):
        #把最高得分初始化为最小的数
        best_score = -float('inf')
        best_children = []
        for c in node.children:
            if c.visits == 0:
                best_children = [c]
                break
            exploit = c.reward / c.visits
            explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
            score = exploit + scalar * explore
            #分数与最高分一样，则加到可选节点中
            if score == best_score:
                best_children.append(c)
            #出现更高的分数，则更新best_score和best_children
            if score > best_score:
                best_children = [c]
                best_score = score
        if len(best_children) == 0:
            return node.parent
        #分数一样的随机选一个
        return random.choice(best_children)

    #随机模拟对弈（不用ucb，完全随机）
    def Stimulate(self, node):
        board = copy.deepcopy(node.state)
        color = node.color
        #记录对弈回合数
        count = 0
        #只要不结束，就一直对弈60次
        while not self.is_end(board):
            action_list = list(node.state.get_legal_actions(color))
            #有合法落子位置，则随机落一个，然后切换庄家，进入下一回合
            if not len(action_list) == 0:
                action = random.choice(action_list)
                board._move(action, color)
                if color == 'X':
                    color = 'O'
                else:
                    color = 'X'
            #无合法落子位置，则进入对方回合，随机落子，然后切换庄家，进入下一回合
            #由于is_end为false，所以不存在双方都无棋可下的局面
            else:
                if color == 'X':
                    color = 'O'
                else:
                    color = 'X'
                action_list = list(node.state.get_legal_actions(color))
                action = random.choice(action_list)
                board._move(action, color)
                if color == 'X':
                    color = 'O'
                else:
                    color = 'X'
            count = count + 1
            #60回合结束（可以调整，平衡好层数和时间关系）
            if count >= 64:
                break

        # 根据60回合后的局面，判断输赢，得分是领先的棋子数
        winner, difference = board.get_winner()
        #平局得分0，自己赢则得分为正，否则为负。
        if winner == 2:
            reward = 0
        elif winner == 1:
            reward = difference
        else:
            reward = -difference

        if self.color == 'X':
            reward = - reward
        return reward
    
    #回溯。回溯时，自己颜色的节点加这个分数，对方颜色的节点减这个分数
    def Back(self, node, reward):
        while node is not None:
            node.visits += 1
            if node.parent==None:
                node.reward-=reward
                break
            if node.parent.color == self.color:
                node.reward += reward
            else:
                node.reward -= reward
            node = node.parent
        return 0
    
    #判断游戏是否结束
    def is_end(self, state):
        # 根据当前棋盘，判断棋局是否终止
        # 如果当前选手没有合法下棋的位子，则切换庄家；如果对方也没有合法的下棋位置，则比赛停止。
        b_list = list(state.get_legal_actions('X'))
        w_list = list(state.get_legal_actions('O'))

        is_over = len(b_list) == 0 and len(w_list) == 0  # 返回值 True/False

        return is_over