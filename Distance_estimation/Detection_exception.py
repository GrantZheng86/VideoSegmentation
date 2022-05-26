class DetectionException(Exception):
    
    def __init__(self, which_procedure):
        super(DetectionException, self).__init__()
        self._procedure = which_procedure

    def __str__(self):
        return 'Exception happen during {}'.format(self._procedure)