# Threading example
import time
import threading
import logging
import random

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


class Task:
    def __init__(self) -> None:
        self.now_ = None
        self.last_ = None
        self.destination = None
        self.orientation = None
        self.picture = None


# receive
    # note: X:Y:ORIENTATION
# mode:1/3:5:N/3:5:S/  .... maze
# mode:2 race


# send location
# note: X:Y:ORIENTATION
# mode:0/3:5:N         #mode 0 refers to updating location
# mode:1/3:5:symbol      #mode 1 refers to updating symbol

class Android:
    def __init__(self) -> None:
        self.msg = None

        pass

    def send_msg(self, msg):
        pass

    def receive_msg(self):  # mode 0 : send list of task ,mode 1 : run maze ,mode 2 : run race
        return True

    def parse_msg(self, msg):  # parse the info to coord or mode
        return 1, []

# receive
# note: X:Y:ORIENTATION
# 1:2:N

# send
#  note: speed:distance:angle
# 1:2:5


class Arduino:
    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        pass

    def send_msg(self, msg):
        pass

    def receive_msg(self):
        logging.debug('Received msg')
        pass

    def update_location(self):
        pass

    def get_location(self):
        return [0, 1]


class Algorithm:
    def __init__(self) -> None:
        pass

    def planpath(self, x, y, destination):
        return True

    def navigation(self):
        return True


class Vision:
    def __init__(self) -> None:
        self.result = "e"
        pass

    def detect(self):
        return True


class RobotMDP28:
    def __init__(self) -> None:
        self.tasks_list = []
        self.mode = 0
        self.cord_x = 0
        self.cord_y = 0
        self.lock = threading.Lock()

        self.android = Android()
        self.algorithm = Algorithm()
        self.arduino = Arduino()
        self.vision = Vision()

    def update_robot_location(self):
        self.lock.acquire()
        location = self.arduino.get_location()
        try:
            logging.debug('Updated robot location')
            self.cord_x = location[0]
            self.cord_y = location[1]
        finally:
            logging.debug('Failed to update robot location')
            self.lock.release()

    def updateAndroid(self):
        self.lock.acquire()
        try:
            logging.debug("Updated android's robot location")
            # send to tablet
        finally:
            logging.debug("Failed to update android's robot location")
            self.lock.release()

    def listenAndroid(self):
        logging.debug('Received msg')
        if self.android.receive_msg():
            mode, coord = self.android.parse_msg(self.android.msg)
            # change mode
            if mode == 1:
                self.mode = 1
                self.addTask(coord)

    def sortTask(self, coordinates):
        pass

    def addTask(self, coordinates):
        """
        plan the sequences of coordinates to travel
        coordinates : a list of [x,y,orientation] coordinates (all the coordinates)
        """
        # TODO
        # sort the task by nearest location
        self.sortTask("HAVENT DONE")

        for i in coordinates:
            task = Task()
            task.now_ = "path_planning"
            task.destination = i[:2]
            task.orientation = i[2]
            self.tasks_list.append(task)

        # return to starting point
        task = Task()
        task.now_ = "path_planning"
        task.destination = 0
        task.destination = 0
        self.tasks_list.append(task)

    def run(self):
        if(self.mode == 0):
            # print("no msg receive")
            return
        if len(self.tasks_list) == 0:
            print("No tasks in the list")
            return

        while len(self.tasks_list) > 0:
            self.taskStep()
        self.mode = 0
        self.reset_state()

    def reset_state(self):
        # reset arduino state
        pass

    def taskStep(self):
        if(self.tasks_list[0].now_ != self.tasks_list[0].last_):
            print("Current task : " + self.tasks_list[0].now_)
        self.tasks_list[0].last_ = self.tasks_list[0].now_
        self.next_state(self.tasks_list[0].now_, self.tasks_list[0])

    def next_state(self, state, param):
        method_name = state
        method = getattr(self, method_name, lambda: 'Invalid')
        return method(param)

    def path_planning(self, param):
        status = self.algorithm.planpath(
            self.cord_x, self.cord_y, param.destination)
        if status:
            self.tasks_list[0].now_ = "navigation"

    def navigation(self, param):
        status = self.algorithm.navigation()
        if status:
            self.tasks_list[0].now_ = "get_vision"

    def get_vision(self, param):
        status = self.vision.detect()
        # update android
        self.android.send_msg(self.vision.result)

        if status:
            self.tasks_list[0].now_ = "end_task"

    def end_task(self, param):
        self.tasks_list.pop(0)

    def run_thread(self):
        while True:
            threading.Thread(target=self.arduino.receive_msg,
                             name="Arduino").start()
            threading.Thread(target=self.listenAndroid,
                             name="Android_receive").start()
            threading.Thread(target=self.update_robot_location,
                             name="locationUpdater").start()
            threading.Thread(target=self.updateAndroid,
                             name="Android_send").start()


if __name__ == '__main__':
    robot = RobotMDP28()
    t1 = threading.Thread(target=robot.run_thread)
    t1.setDaemon(True)
    t1.start()
    while True:

        robot.run()
