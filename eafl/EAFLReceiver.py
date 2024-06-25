import time


class EAFLReceiver:
    def __init__(self, config):
        self.config = config

    def receive(self, queue, nums):
        # 第i组/层全都上传完成
        for i in range(len(nums)):
            while queue[i].qsize() < nums[i]:
                time.sleep(0.1)
