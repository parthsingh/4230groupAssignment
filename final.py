from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt

def split_matrix_by_rows(matrix, num_parts):
    num_rows_per_part = matrix.shape[0] // num_parts
    matrix_parts = []
    for i in range(num_parts):
        start_row = i * num_rows_per_part
        end_row = (i + 1) * num_rows_per_part if i < num_parts - 1 else None
        matrix_parts.append(matrix[start_row:end_row])
    return matrix_parts

def split_matrix_by_columns(matrix, num_parts):
    num_cols_per_part = matrix.shape[1] // num_parts
    matrix_parts = []
    for i in range(num_parts):
        start_col = i * num_cols_per_part
        end_col = (i + 1) * num_cols_per_part if i < num_parts - 1 else None
        matrix_parts.append(matrix[:, start_col:end_col])
    return matrix_parts

def matrix_multiply(A_parts, B_parts):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    local_n = len(A_parts[0])
    local_C = np.zeros((local_n, local_n))
    
    # From pseudo code provided on canvas 
    # 1D matrix multiplication
    for t in range(size):
        SPROC = (rank + t) % size
        RPROC = (rank - t + size) % size
        B_part_to_send = np.ascontiguousarray(B_parts[RPROC])
        # B_part_to_send = B_parts[RPROC]
        B_part_received = np.empty_like(B_part_to_send)
        comm.Sendrecv(B_part_to_send, dest=SPROC, recvbuf=B_part_received, source=RPROC)
        for i in range(local_n):
            for j in range(local_n):
                for k in range(len(B_part_received)):
                    local_C[i, j] += np.dot(A_parts[SPROC][i, k], B_part_received[k, j])
    
    # Gather local results to root process
    C_parts = comm.gather(local_C, root=0)
    
    return C_parts

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Define the values of N and P for the experiments
    N_values = [5000] # 100, 1000, 5000, 10000
    P_values = [1, 2, 4, 8]
    
    # Dictionary to store runtimes for different values of N and P
    runtimes = {}
    
    for N in N_values:
        runtimes[N] = []
        for P in P_values:
            start_time = time.time()
            # Create matrix A(NxN) and B(NxN)
            A = np.random.randint(10, size=(N, N))
            B = np.random.randint(10, size=(N, N))
            
            A = comm.bcast(A, root=0)
            B = comm.bcast(B, root=0)
            
            # Split the matrixes to submatrices
            A_parts = split_matrix_by_rows(A, P)
            B_parts = split_matrix_by_columns(B, P)
            
            # Do matrix multplixation C(NxN) = A(NxN) * B(NxN)
            C_parts = matrix_multiply(A_parts, B_parts)
            
            # Calculate the total time it took to compute the matrix multiplication
            end_time = time.time()
            runtime = end_time - start_time
            runtimes[N].append(runtime)
    
    # Print the runtimes in a table
    print("Runtime P=1 P=2 P=4 P=8 (seconds)")
    for N in N_values:
        print(f"n = {N}: ", end="")
        for runtime in runtimes[N]:
            print(f"{runtime:.6f} | ", end=" ")
        print()
        
   # Calculate speedup
    speedups = {}
    for N in N_values:
        runtime_P1 = runtimes[N][0]  # Runtime when P=1
        speedups[N] = [runtime_P1 / runtime for runtime in runtimes[N][1:]]
    print("Speedups:", speedups)

        
   # Plot speedup
    for N in N_values:
        plt.plot(P_values[1:], speedups[N], marker='o', label=f'n = {N}')

    plt.xlabel('Number of Processors (P)')
    plt.ylabel('Speedup (S)')
    plt.title('Speedup vs. Number of Processors')
    plt.grid(True)
    plt.legend()
    plt.show() 
    plt.savefig("image.png")
    
if __name__ == "__main__":
    main()

