from envs import settings, Manager
from envs.Entity import Entity
from envs.Task import Task
from envs.UAV import UAV


class BigTask(Entity):
    def __init__(self, name:str, small_task_num:int):
        super().__init__(name)
        self.small_task_num = small_task_num

        self.color = (0.5, 1, 0.5)
        self.size = 0.4
        self.is_take = False
        self.small_task_list = []

        self.done_small_tasks = 0
        self.remain_time = settings.task_alive_time
        self.take_time = None
        self.renew = False

        self.belong_uav = None
        self.is_fail = False
        self.is_done = False
        self.status = []

        self.take_consume_time = 0

        for i in range(small_task_num):
            task = Task( self.remain_time, name + '_子任务_' + str(i))
            task.belong_big_task = self
            self.small_task_list.append(task)


    def set_position(self, x, y):
        super().set_position(x, y)
        for task in self.small_task_list:
            task.x, task.y = x, y



    @property
    def get_success_num(self):
        return sum(task.is_done for task in self.small_task_list)

    @property
    def is_success(self):
        return self.get_success_num == self.small_task_num

    def take(self, uav: UAV):
        self.status.append(f'剩余时间{self.remain_time}, 将任务交给{uav.name}')
        if not self.is_take:
            self.is_take = True
            self.take_time = self.remain_time
            for task in self.small_task_list:
                
                task.remain_time = self.remain_time
                task.take(uav)
            self.take_consume_time = self.small_task_list[0].take_consume_time
        else:
            raise NotImplementedError(f"当前任务{self.name}已被认领，请勿重复take")

    def isExpire(self):
        if self.remain_time <= Manager.get_min_execute_time():
            return True
        return False

    def all_done(self):
        return self.done_small_tasks == self.small_task_num

    def get_task_success_num(self):
        return len([task for task in self.small_task_list if task.is_done])

    def show_small_tasks_info(self):
        print(f'  {self.name}, {self.get_status()}')
        for task in self.small_task_list:
            print(f'     {task.status_info()}')

    def get_status(self):
        msg = '未接受'
        if self.is_take:
            msg = f'接受时间 {self.take_time}'

        if self.is_success:
            return f"成功完成 剩余时间{self.remain_time}, {msg}, 差值:{self.take_time - self.remain_time}"

        if self.is_fail:
            return f"失败, {msg}, 任务完成率:{self.get_success_num}"

        return msg


    def fail(self):
        self.status.append(f'剩余时间{self.remain_time}, {self.name} 执行失败')
        self.is_fail = True
        for task in self.small_task_list:
            task.remain_time = self.remain_time
            task.fail()

    def set_small_tasks_position(self, x, y):
        for small_task in self.small_task_list:
            small_task.x, small_task.y = x, y

    def get_success_small_tasks_trans_num(self):
        return sum(len(small_task.trans_path) for small_task in self.small_task_list if small_task.is_done)


