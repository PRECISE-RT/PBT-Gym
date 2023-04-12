import pbtgym

gym = pbtgym.PbtGym()

gym.dose_model = pbtgym.BitWiseDose()

gym.dose_model.apply_dose()