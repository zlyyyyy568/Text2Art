import threading


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None


class MyQueue:
    """
    多线程安全的队列
    """

    def __init__(self):
        self.head = None  # 对头指针
        self.tail = None  # 队尾指针
        # 创建了一个可重入锁，可重入锁允许同一个线程多次获取该锁而不会导致死锁，又能防止多个线程同时修改共享资源，从而保证线程安全性。
        self.lock = threading.RLock()

    def isEmpty(self):
        """
        队列是否为空
        """

        # 使用with self.lock获取锁，以确保同一时间只有一个线程可以执行被保护的代码块
        with self.lock:
            return self.head is None

    def enqueue(self, data):
        """
        入队
        """

        newNode = Node(data)

        with self.lock:
            if self.tail is None:
                self.head = self.tail = newNode
            else:
                self.tail.next = newNode
                self.tail = newNode

    def dequeue(self):
        """
        出队
        """

        if self.isEmpty():
            raise Exception("链表为空，无法出队")

        with self.lock:
            dequeueNode = self.head
            if self.tail is dequeueNode:
                # 若出队节点为队列中最后一个节点
                self.tail = None
            self.head = dequeueNode.next
            return dequeueNode

    def peek(self):
        """
        查看队头第一个元素
        """

        if self.isEmpty():
            return None

        with self.lock:
            return self.head.data

    def remove(self, data):
        """
        将data从队列中移除
        """

        with self.lock:
            # 如果队列第一个元素就是我们需要删除的元素
            if self.head.data == data:
                self.dequeue()
                return

            # 通过上面的if语句判断的队列，其大小>=2，因此删除后不需要考虑队列是否为空的情况（即tail不可能为None)
            # prev_p指向要删除节点的前一个节点
            prev_p = self.head

            while prev_p.next is not None and prev_p.next.data != data:
                prev_p = prev_p.next

            if prev_p.next is None:
                raise Exception("未找到需要删除的元素")

            removeNode = prev_p.next
            prev_p.next = removeNode.next
            if self.tail is removeNode:
                # 若需要删除的节点为队列中最后一个节点
                self.tail = prev_p

            del removeNode

    def showList(self):
        with self.lock:
            ptr = self.head

            while ptr is not None:
                print(ptr.data, end=' ')
                ptr = ptr.next
        print()

    def __iter__(self):
        """
        使队列可迭代
        """

        current_node = self.head
        while current_node:
            yield current_node.data
            current_node = current_node.next


if __name__ == "__main__":
    MyQueue = MyQueue()

    MyQueue.enqueue(1)
    MyQueue.enqueue(2)
    MyQueue.enqueue(3)

    MyQueue.showList()

    MyQueue.dequeue()
    MyQueue.showList()
    MyQueue.dequeue()
    MyQueue.dequeue()
    MyQueue.enqueue(4)
    MyQueue.enqueue(5)
    MyQueue.enqueue(6)
    MyQueue.showList()

    MyQueue.remove(4)
    MyQueue.showList()
    MyQueue.remove(6)
    MyQueue.showList()
    MyQueue.remove(5)
    MyQueue.showList()
    print('finish!')
