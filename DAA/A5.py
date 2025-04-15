import random

def deterministic_quick_sort(arr):
  if len(arr) <= 1:
    return arr

  pivot = arr[0]
  left = []
  right = []

  for item in arr[1:]:
    global comparisons
    comparisons += 1  
    if item < pivot:
      left.append(item)
    else:
      right.append(item)

  return deterministic_quick_sort(left) + [pivot] + deterministic_quick_sort(right)


def randomized_quick_sort(arr):
  if len(arr) <= 1:
    return arr

  pivot_idx = random.randint(0, len(arr) - 1)
  pivot = arr[pivot_idx]

  left = []
  right = []

  for idx, item in enumerate(arr):
    global comparisons
    comparisons += 1  
    if idx == pivot_idx:
      continue
    if item < pivot:
      left.append(item)
    else:
      right.append(item)

  return randomized_quick_sort(left) + [pivot] + randomized_quick_sort(right)

def count_comparisons(arr, sorting_algorithm):
  global comparisons
  comparisons = 0

  sorted_arr = sorting_algorithm(arr)
  return comparisons, sorted_arr

if __name__ == "__main__":

  input_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

  deterministic_comparisons, deterministic_sorted = count_comparisons(input_list.copy(), deterministic_quick_sort)
  print("Deterministic QuickSort:")
  print("Comparisons:", deterministic_comparisons)
  print("Sorted Array:", deterministic_sorted)

  randomized_comparisons, randomized_sorted = count_comparisons(input_list.copy(), randomized_quick_sort)
  print("\nRandomized QuickSort:")
  print("Comparisons:", randomized_comparisons)
  print("Sorted Array:", randomized_sorted)
