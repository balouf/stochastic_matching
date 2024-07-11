from numba.experimental import jitclass
from numba import typeof, int64
from numba.typed import Dict
import numpy as np
import os


if os.environ.get("NUMBA_DISABLE_JIT") == "1":
    vq_type = dict
else:
    vq_type = typeof(Dict.empty(key_type=int64, value_type=int64))

infinity = 2 ** 63 - 1
infinity_type = typeof(infinity)


@jitclass
class MultiQueue:
    """
    A cruder, faster implementation of FullMultiQueue. Queues must be explicitly bounded on initialisation.

    Parameters
    ----------
    n: :class:`int`
        Number of classes.

    Examples
    --------

    Let's populate a multiqueue and play with it.

    >>> queues = MultiQueue(4, max_queue=4)
    >>> _ = [queues.add(i, s) for i, s in [(0, 2), (0, 5), (1, 4), (3, 0), (3, 1), (3, 6)]]
    >>> [queues.size(i) for i in range(4)]
    [2, 1, 0, 3]

    The age of the oldest element of an empty queue is infinity.

    >>> [queues.oldest(i) for i in range(4)]
    [2, 4, 9223372036854775807, 0]
    >>> queues.add(3, 10)
    >>> [queues.size(i) for i in range(4)]
    [2, 1, 0, 4]

    Trying to go above the max queue raises an error (item is not added, queue stays viable):

    >>> queues.add(3, 10)
    Traceback (most recent call last):
    ...
    OverflowError: Max queue size reached.

    So does trying to pop something empty (and queue stays viable).

    >>> queues.pop(2)
    Traceback (most recent call last):
    ...
    ValueError: Try to pop an empty queue

    Let's play a bit more.

    >>> _ = [queues.pop(i) for i in [0, 1, 3, 3]]
    >>> [queues.size(i) for i in range(4)]
    [1, 0, 0, 2]
    >>> [queues.oldest(i) for i in range(4)]
    [5, 9223372036854775807, 9223372036854775807, 6]
    >>> for a in range(22, 30):
    ...     print(f"Add {a} to class 3.")
    ...     queues.add(3, a)
    ...     print(f"Oldest of class 3 is {queues.oldest(3)}, we pop it.")
    ...     queues.pop(3)
    Add 22 to class 3.
    Oldest of class 3 is 6, we pop it.
    Add 23 to class 3.
    Oldest of class 3 is 10, we pop it.
    Add 24 to class 3.
    Oldest of class 3 is 22, we pop it.
    Add 25 to class 3.
    Oldest of class 3 is 23, we pop it.
    Add 26 to class 3.
    Oldest of class 3 is 24, we pop it.
    Add 27 to class 3.
    Oldest of class 3 is 25, we pop it.
    Add 28 to class 3.
    Oldest of class 3 is 26, we pop it.
    Add 29 to class 3.
    Oldest of class 3 is 27, we pop it.
    >>> [queues.size(i) for i in range(4)]
    [1, 0, 0, 2]
    >>> [queues.oldest(i) for i in range(4)]
    [5, 9223372036854775807, 9223372036854775807, 28]
    """
    queues: int64[:, :]
    heads: int64[:]
    tails: int64[:]
    infinity: infinity_type
    max_queue: infinity_type

    def __init__(self, n, max_queue=10000):
        self.infinity = 2 ** 63 - 1
        self.queues = np.ones((n, max_queue), dtype=np.int64)
        self.queues[:] = self.infinity
        self.heads = np.zeros(n, dtype=np.int64)
        self.tails = np.zeros(n, dtype=np.int64)
        self.max_queue = max_queue

    def add(self, item_class, stamp):
        """
        Parameters
        ----------
        item_class: :class:`int`
            Class of the item.
        stamp: :class:`int`
            Item stamp to add to the queue of the item class.

        Returns
        -------
        None

        """
        if self.queues[item_class, self.tails[item_class]] != self.infinity:
            raise OverflowError("Max queue size reached.")
        self.queues[item_class, self.tails[item_class]] = stamp
        self.tails[item_class] = (self.tails[item_class] + 1) % self.max_queue

    def oldest(self, item_class):
        """
        Parameters
        ----------
        item_class: :class:`int`
            Class of the item.

        Returns
        -------
        :class:`int`
            Stamp of the oldest item (or infinity if the queue is empty).
        """
        return self.queues[item_class, self.heads[item_class]]

    def pop(self, item_class):
        """
        Remove the oldest item from given class. Item stamp is NOT returned (use oldest beforehand if you need access).

        Parameters
        ----------
        item_class: :class:`int`
            Class of the item to remove.

        Returns
        -------
        None
        """

        if self.queues[item_class, self.heads[item_class]] == self.infinity:
            raise ValueError("Try to pop an empty queue")
        self.queues[item_class, self.heads[item_class]] = self.infinity
        self.heads[item_class] = (self.heads[item_class] + 1) % self.max_queue

    def size(self, item_class):
        if self.queues[item_class, self.tails[item_class]] == self.infinity:
            return (self.tails[item_class] - self.heads[item_class]) % self.max_queue
        return self.max_queue


@jitclass
class FullMultiQueue:
    """
    A simple management of multi-class FCFM that supports add to tail
    and pop from head over multiple item classes.
    Each item must be characterized by a UNIQUE non-negative int (typically a timestamp).

    Parameters
    ----------
    n: :class:`int`
        Number of classes.

    Examples
    --------

    Let's populate a multiqueue.

    >>> queues = FullMultiQueue(4)
    >>> for cl, stamp in [(0, 2), (0, 5), (1, 4), (3, 0), (3, 1), (3, 6)]:
    ...     queues.add(cl, stamp)

    Oldest items per class (infinity indicates an empty queue):

    >>> [queues.oldest(i) for i in range(4)]
    [2, 4, 9223372036854775807, 0]

    Let's remove some item. Note that pop returns nothing.

    >>> [queues.pop(i) for i in [0, 1, 3, 3]]
    [None, None, None, None]

    Oldest items per class (infinity indicates an empty queue):

    >>> [queues.oldest(i) for i in range(4)]
    [5, 9223372036854775807, 9223372036854775807, 6]

    Note that trying to pop from an empty queue will raise an error:

    >>> queues.pop(1)
    Traceback (most recent call last):
    ...
    KeyError: 9223372036854775807
    """
    tails: int64[:]
    vq: vq_type
    infinity: infinity_type

    def __init__(self, n):
        self.infinity = 2 ** 63 - 1
        self.tails = np.array([-1 - e for e in range(n)], dtype=np.int64)
        self.vq = {t: self.infinity for t in self.tails}

    def add(self, item_class, stamp):
        """
        Parameters
        ----------
        item_class: :class:`int`
            Class of the item.
        stamp: :class:`int`
            Item stamp to add to the queue of the item class.

        Returns
        -------
        None

        """
        self.vq[self.tails[item_class]] = stamp  # tail points to a real age
        self.vq[stamp] = self.infinity  # new age points to infinity
        self.tails[item_class] = stamp  # Move up tail

    def oldest(self, item_class):
        """
        Parameters
        ----------
        item_class: :class:`int`
            Class of the item.

        Returns
        -------
        :class:`int`
            Stamp of the oldest item (or infinity if the queue is empty).
        """
        return self.vq[-1 - item_class]

    def pop(self, item_class):
        """
        Remove the oldest item from given class. Item stamp is NOT returned (use oldest beforehand if you need access).

        Parameters
        ----------
        item_class: :class:`int`
            Class of the item to remove.

        Returns
        -------
        None
        """
        stamp = self.oldest(item_class)  # get oldest item of item_class
        self.vq[-1 - item_class] = self.vq[stamp]  # head points to next item
        del self.vq[stamp]  # remove oldest item
        if stamp == self.tails[item_class]:  # if queue is empty, plug tail to head
            self.tails[item_class] = -1 - item_class
