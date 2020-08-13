"""
This file contains helper functions to verify that flags are given permissible values. 
"""

def list_contains_only_integers(L):
  """
  A function that returns True if the input list contains only string representations of digits.

  Args:
  L: A list. Elements must be strings, and if any element is not a string representation of a digit,
     the function returns False. 
  """
  if type(L) != list:
    raise ValueError(f"The given input {L} is not a list")
  for elm in L:
    if type(elm) != str:
      raise ValueError(f"Element {elm} is not a string")
    if not elm.isdigit():
      return False
  return True

