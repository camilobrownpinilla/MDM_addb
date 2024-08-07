import nimblephysics as nimble 
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os

def is_file_empty(root, filename):
    filepath = os.path.join(root, filename)
    return os.stat(filepath).st_size == 0


class InputDataKeys:
    # These are the joint quantities for the joints that we are observing
    POS = 'pos'
    VEL = 'vel'
    ACC = 'acc'

    COM_POS = 'comPos'
    COM_VEL = 'comVel'
    COM_ACC = 'comAcc'

class OutputDataKeys:
    # TAU = 'tau'

    # These are enough to compute ID
    
    # GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME = 'groundContactWrenchesInRootFrame'
    # RESIDUAL_WRENCH_IN_ROOT_FRAME = 'residualWrenchInRootFrame'

    # These are various other things we might want to predict
    CONTACT = 'contact'
    COM_ACC_IN_ROOT_FRAME = 'comAccInRootFrame'
    COM_POS = 'comPos'
    COM_VEL = 'comVel'
    COM_ACC = 'comAcc'
    ACC = 'acc'
    POS = 'pos'
    VEL = 'vel'

    ROOT_ANGULAR_ACC = 'rootAngularAccInRootFrame'
    ROOT_ANGULAR_VEL = 'rootAngularVelInRootFrame'
    ROOT_EULER_HISTORY = 'rootEulerHistoryInRootFrame'
    # GROUND_CONTACT_COPS_IN_ROOT_FRAME = 'groundContactCenterOfPressureInRootFrame'
    # GROUND_CONTACT_TORQUES_IN_ROOT_FRAME = 'groundContactTorqueInRootFrame'
    # GROUND_CONTACT_FORCES_IN_ROOT_FRAME = 'groundContactForceInRootFrame'

class AddBiomechanicsDataset(Dataset):
    stride: int
    data_path: str
    window_size: int
    geometry_folder: str
    device: torch.device
    dtype: torch.dtype
    subject_paths: List[str]
    subjects: List[nimble.biomechanics.SubjectOnDisk]
    windows: List[Tuple[int, int, int]]  # Subject, trial, start_frame
    num_dofs: int
    num_joints: int
    contact_bodies: List[str]
    # For each subject, we store the skeleton and the contact bodies in memory, so they're ready to use with Nimble
    skeletons: List[nimble.dynamics.Skeleton]
    skeletons_contact_bodies: List[List[nimble.dynamics.BodyNode]]
    subject_indices: Dict[str, int]

    def __init__(self,
                 data_path: str,
                 window_size: int,
                 geometry_folder: str,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float64,
                 testing_with_short_dataset: bool = True,
                 stride: int = 1,
                 output_data_format: str = 'all_frames',
                 skip_loading_skeletons: bool = True):
        self.stride = stride
        self.output_data_format = output_data_format
        self.subject_paths = []
        self.subjects = []
        self.window_size = window_size
        self.geometry_folder = geometry_folder
        self.device = device
        self.dtype = dtype
        self.windows = []
        self.contact_bodies = []
        self.skeletons = []
        self.skeletons_contact_bodies = []

        if os.path.isdir(data_path):
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith(".b3d") and "vander" not in file.lower():
                        self.subject_paths.append(os.path.join(root, file))
        else:
            assert data_path.endswith(".b3d")
            self.subject_paths.append(data_path)

        if testing_with_short_dataset:
            self.subject_paths = self.subject_paths[2:3]
        self.subject_indices = {subject_path: i for i, subject_path in enumerate(self.subject_paths)}

        # Walk the folder path, and check for any with the ".b3d" extension (indicating that they are
        # AddBiomechanics binary data files)
        if len(self.subject_paths) > 0:
            # Create a subject object for each file. This will load just the header from this file, and keep that
            # around in memory
            subject = nimble.biomechanics.SubjectOnDisk(
                self.subject_paths[0])
            # Get the number of degrees of freedom for this subject
            self.num_dofs = subject.getNumDofs()
            # Get the number of joints for this subject
            self.num_joints = subject.getNumJoints()
            # Get the contact bodies for this subject, and put them into a consistent order for the dataset
            contact_bodies = subject.getGroundForceBodies()
            for body in contact_bodies:
                if body == 'pelvis':
                    continue
                if body not in self.contact_bodies:
                    self.contact_bodies.append(body)

        for i, subject_path in enumerate(self.subject_paths):
            # Add the skeleton to the list of skeletons
            subject = nimble.biomechanics.SubjectOnDisk(subject_path)
            if not skip_loading_skeletons:
                print('Loading skeleton ' + str(i + 1) + '/' + str(
                    len(self.subject_paths)) + f' for subject {subject_path}')
                skeleton = subject.readSkel(subject.getNumProcessingPasses() - 1, geometry_folder)
                self.skeletons.append(skeleton)
                self.skeletons_contact_bodies.append([skeleton.getBodyNode(body) for body in self.contact_bodies])
            self.subjects.append(subject)
            # Prepare the list of windows we can use for training
            for trial_index in range(subject.getNumTrials()):
                trial_length = subject.getTrialLength(trial_index)
                probably_missing: List[bool] = [reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason
                                                in subject.getMissingGRF(trial_index)]
                for window_start in range(max(trial_length - self.window_size - 1, 0)):
                    if not any(probably_missing[window_start:window_start + self.window_size:self.stride]):
                        assert window_start + self.window_size < trial_length
                        self.windows.append((i, trial_index, window_start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int]:
        subject_index, trial, window_start = self.windows[index]

        # Read the frames from disk
        subject = self.subjects[subject_index]
        frames: nimble.biomechanics.FrameList = subject.readFrames(trial,
                                                                   window_start,
                                                                   self.window_size // self.stride,
                                                                   stride=self.stride,
                                                                   includeSensorData=False,
                                                                   includeProcessingPasses=True)
        assert (len(frames) == self.window_size // self.stride)

        first_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[0] for frame in frames] #Kinematics
        output_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[-1] for frame in frames]

        input_dict: Dict[str, torch.Tensor] = {}
        label_dict: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            input_dict[InputDataKeys.POS] = torch.row_stack([
                torch.tensor(p.pos, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.VEL] = torch.row_stack([
                torch.tensor(p.vel, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.ACC] = torch.row_stack([
                torch.tensor(p.acc, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.COM_POS] = torch.row_stack([
                torch.tensor(p.comPos, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.COM_VEL] = torch.row_stack([
                torch.tensor(p.comVel, dtype=self.dtype).detach() for p in first_passes
            ])
            input_dict[InputDataKeys.COM_ACC] = torch.row_stack([
                torch.tensor(p.comAcc, dtype=self.dtype).detach() for p in first_passes
            ])

            # The output dictionary contains a single frame, the last frame in the window if output_data_format is 2d
            # else it contains outputs for all the frames in first_passes
            mass = subject.getMassKg()
            start_index = 0 if self.output_data_format == 'all_frames' else -1
            label_dict[OutputDataKeys.CONTACT] = torch.row_stack([
                torch.tensor(p.contact, dtype=self.dtype).detach() for p in output_passes[start_index:]
            ])
            # label_dict[OutputDataKeys.COM_ACC_IN_ROOT_FRAME] = torch.row_stack([
            #     torch.tensor(p.comAccInRootFrame, dtype=self.dtype).detach() for p in output_passes[start_index:]
            # ])
            label_dict[OutputDataKeys.COM_ACC] = torch.row_stack([
                torch.tensor(p.comAcc, dtype=self.dtype).detach() for p in output_passes[start_index:]
            ])
            label_dict[OutputDataKeys.COM_POS] = torch.row_stack([
                torch.tensor(p.comPos, dtype=self.dtype).detach() for p in output_passes[start_index:]
            ])
            label_dict[OutputDataKeys.COM_VEL] = torch.row_stack([
                torch.tensor(p.comVel, dtype=self.dtype).detach() for p in output_passes[start_index:]
            ])
            label_dict[OutputDataKeys.ACC] = torch.row_stack([
                torch.tensor(p.acc, dtype=self.dtype).detach() for p in output_passes[start_index:]
            ])
            label_dict[OutputDataKeys.POS] = torch.row_stack([
                torch.tensor(p.pos, dtype=self.dtype).detach() for p in output_passes[start_index:]
            ])
            label_dict[OutputDataKeys.VEL] = torch.row_stack([
                torch.tensor(p.vel, dtype=self.dtype).detach() for p in output_passes[start_index:]
            ])

            """
            TEST COMMENT OUT
            """
            # label_dict[OutputDataKeys.ROOT_ANGULAR_ACC] = torch.row_stack([
            #     torch.tensor(p.rootAngularAccInRootFrame, dtype=self.dtype).detach() for p in output_passes[start_index:]
            # ])
            # label_dict[OutputDataKeys.ROOT_ANGULAR_VEL] = torch.row_stack([
            #     torch.tensor(p.rootAngularVelInRootFrame, dtype=self.dtype).detach() for p in output_passes[start_index:]
            # ])
            # label_dict[OutputDataKeys.ROOT_EULER_HISTORY] = torch.row_stack([
            #     torch.tensor(p.rootEulerHistoryInRootFrame, dtype=self.dtype).detach() for p in output_passes[start_index:]
            # ])
            
            
        # Convert the frames to a dictionary of matrices, where columns are timesteps and rows are degrees of freedom / dimensions
        # (the DataLoader will then convert this to a batched tensor)
        # print(f"{numpy_output_dict[OutputDataKeys.CONTACT_FORCES]=}")
        # ###################################################
        # # Plotting
        # import matplotlib.pyplot as plt
        # x = np.arange(self.window_size)
        # # plotting each row
        # for i in range(len(self.input_dofs)):
        #     # plt.plot(x, numpy_input_dict[InputDataKeys.POS][i, :], label='pos_'+self.input_dofs[i])
        #     plt.plot(x, numpy_input_dict[InputDataKeys.VEL][i, :], label='vel_' + self.input_dofs[i])
        #     plt.plot(x, numpy_input_dict[InputDataKeys.ACC][i, :], label='acc_' + self.input_dofs[i])
        # for i in range(3):
        #     plt.plot(x, numpy_input_dict[InputDataKeys.COM_ACC][i, :], label='com_acc_' + str(i))
        # # Add the legend outside the plot
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.show()
        # ###################################################

        # Return the input and output dictionaries at this timestep, as well as the skeleton pointer

        return input_dict, label_dict, subject_index, trial
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['subjects']
        del state['skeletons']
        del state['skeletons_contact_bodies']
        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        self.subjects = []
        print('Unpickling AddBiomechanicsDataset copy in reader worker thread')
        # Create the non picklable SubjectOnDisk objects. Skip loading the skeletons and contact bodies, since these
        # are not used in the reader worker threads.
        for i, subject_path in enumerate(self.subject_paths):
            self.subjects.append(nimble.biomechanics.SubjectOnDisk(subject_path))