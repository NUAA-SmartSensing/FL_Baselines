from scheduler.SyncScheduler import SyncScheduler


class ScaffoldScheduler(SyncScheduler):
    def customize_download(self):
        self.download_item("all", "control", self.global_var['control'])
