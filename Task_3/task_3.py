import numpy as np

def input_matrix(name="A"):
    print(f"\n--- Input Matrix {name} ---")
    try:
        rows = int(input(f"Enter number of rows for {name}: "))
        cols = int(input(f"Enter number of columns for {name}: "))
    except ValueError:
        print("Error: Dimensions must be integers.")
        return None

    print(f"Enter the {rows}x{cols} matrix row by row (space-separated numbers):")
    matrix_data = []
    
    for i in range(rows):
        while True:
            row_input = input(f"Row {i+1}: ").strip().split()
            try:
                row = [float(x) for x in row_input]
                if len(row) != cols:
                    print(f"Error: Expected {cols} values, got {len(row)}. Try again.")
                    continue
                matrix_data.append(row)
                break
            except ValueError:
                print("Error: Invalid input. Please enter numbers only. Try again.")
                
    return np.array(matrix_data)

def display_result(operation_name, result):
    print(f"\n=== Result: {operation_name} ===")
    if isinstance(result, np.ndarray):
        print(np.array2string(result, precision=4, separator='  ', suppress_small=True))
    else:
        print(f"{result:.4f}")
    print("=========================")

def main():
    print("Welcome to the Interactive Matrix Operations Tool!")
    
    while True:
        print("\nSelect an operation:")
        print("1. Addition (A + B)")
        print("2. Subtraction (A - B)")
        print("3. Multiplication (A * B)")
        print("4. Transpose (A^T)")
        print("5. Determinant (|A|)")
        print("6. Exit")
        
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == '6':
            print("Exiting tool. Goodbye!")
            break
            
        if choice in ['1', '2', '3']:
            mat_a = input_matrix("A")
            if mat_a is None: continue
            
            mat_b = input_matrix("B")
            if mat_b is None: continue
            
            try:
                if choice == '1':
                    display_result("Addition", np.add(mat_a, mat_b))
                elif choice == '2':
                    display_result("Subtraction", np.subtract(mat_a, mat_b))
                elif choice == '3':
                    display_result("Matrix Multiplication", np.matmul(mat_a, mat_b))
            except ValueError as e:
                print(f"\nMathematical Error: {e}")
                print("Please check your matrix dimensions!")
                
        elif choice in ['4', '5']:
            mat_a = input_matrix("A")
            if mat_a is None: continue
            
            try:
                if choice == '4':
                    display_result("Transpose", np.transpose(mat_a))
                elif choice == '5':
                    if mat_a.shape[0] != mat_a.shape[1]:
                        print("\nMathematical Error: Determinant requires a square matrix (NxN).")
                    else:
                        display_result("Determinant", np.linalg.det(mat_a))
            except Exception as e:
                print(f"\nError: {e}")
                
        else:
            print("Invalid choice. Please select a number from 1 to 6.")

if __name__ == "__main__":
    main()