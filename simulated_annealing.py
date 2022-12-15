"""
Code for CM3109 Coursework by:
Matthew Battagel
C1946077
"""


import sys
import time
import math
import random
import concurrent.futures


class SimulatedAnnealing():
    """
    Contains all code for performing the simulated annealing
    """

    def __init__(self):
        """
        Set the SA parameters and collect file data
        """
        self.log = []
        # Set SA Parameters
        self.initial_temperature: float = 10
        self.temperature_length: int = 100
        self.temperature_multiplier: float = 0.95
        self.num_non_improve: int = 250
        self.run_number: int = 5

        self._collect_user_input()

    def _collect_user_input(self):
        # Get file path containing data
        if len(sys.argv) < 2:
            print("Please enter the file path and try again")
            sys.exit()
        file_path: str = sys.argv[1]

        # Read data from file
        raw_data: list[str] = self._read_data(file_path)
        data = self._chunk_data(raw_data)
        self.data_length = data[0]
        self.drivers = data[1]
        self.meta = data[2]
        self.matches = data[3]
        self.score_matrix: list[list[int]] = self._gen_score_matrix()

    def _read_data(self, path: str) -> list[str]:
        """
        Read in the data from the file path and return list
        """
        with open(path, "r", encoding="utf-8") as file:
            data = [line.strip() for line in file]
            return data

    def _chunk_data(self, unchunked_data: list[str]) -> tuple[int,
                                                              list[str],
                                                              str,
                                                              list[str]]:
        """
        Split the raw input data into useful formats.
        """
        length_chunk = int(unchunked_data[0])
        ids_chunk = unchunked_data[1:47]
        meta_chunk = unchunked_data[47]
        matches_chunk = unchunked_data[48:]

        return length_chunk, ids_chunk, meta_chunk, matches_chunk

    def _gen_initial_soluion(self) -> list[int]:
        """
        Generate a list containing numbers from 1 to ranking length.
        """
        gen_init_list: list = []
        count: int = 1
        while count <= self.data_length:
            gen_init_list.append(count)
            count += 1
        return gen_init_list

    def _cooling_ratio(self, temp: float) -> float:
        """
        Cooling ratio for reducing temperature over iterations
        """
        return self.temperature_multiplier * temp

    def _gen_score_matrix(self) -> list[list[int]]:
        """
        Use the matches from the supplied file to create a matrix of all scores
        with size LENGTH * LENGTH

        Non-symmetrical as the score is only counted for the winner
        """
        # Generate empty matrix with size LENGHT * LENGTH
        score_matrix: list[list[int]] = [[0 for _ in range(self.data_length)]
                                         for _ in range(self.data_length)]
        # For each match add the weighting to matrix
        match: str
        for match in self.matches:
            match_split: list[str] = match.split(",")
            weight: int = int(match_split[0])
            player1: int = int(match_split[1])
            player2: int = int(match_split[2])

            score_matrix[player1-1][player2-1] = weight

        # Add -1 for invalid values for error checking
        for x_index, _ in enumerate(score_matrix):
            for y_index, _ in enumerate(score_matrix):
                if x_index == y_index:
                    score_matrix[y_index][x_index] = -1

        return score_matrix

    def _debug_print_matrix(self, matrix: list[list[int]]):
        """
        Prints any size square matrix

        Useful for debugging
        """
        print('\n'.join('\t'.join(map(str, row)) for row in matrix))
        print("Printed Matrix")
        print(f"Length: {len(matrix)}")

    def _gen_random_indexes(self) -> tuple[int, int]:
        """
        Generate two random indexes in size order that are not adjacent
        """
        # Number between 0 (inclusive) - 1 (exclusive)
        random1: float = random.random()
        random2: float = random.random()
        # Find number of edges
        edges: int = self.data_length + 1
        # Match random value with corresponding edge
        edge1: int = int(random1 / (1/edges))
        # Find edges to the left of edge1
        left_edges = edge1 - 2
        # Calculate number of edges left
        edges_left: int = self.data_length - 2
        # Match second random value with corresponding edge
        index = int(random2 / (1/edges_left))
        edge2: int
        # Find if edge2 is on left or right
        if index <= left_edges:
            edge2 = index
        else:
            edge2 = index + 3
        # Find largest edge
        if edge1 > edge2:
            return edge2, edge1
        return edge1, edge2

    def _calc_kemeny_score(self, solution: list[int]) -> int:
        """
        Calculate the kemeny score of a list with respect to the tournament
        data.

        For each of the drivers in the ranking find all drivers lower in
        ranking and compare their matches. If there is a disagreement with
        the ranking then add the waiting to the kemeny score
        """
        kemeny_score: int = 0
        target_index: int
        driver1: int
        for target_index, driver1 in enumerate(solution):
            remaining_index: int
            for remaining_index in range(target_index + 1, len(solution)):
                driver2: int = solution[remaining_index]
                disagreement: int = self.score_matrix[driver2-1][driver1-1]
                kemeny_score += disagreement
        return kemeny_score

    def _cost_function(self,
                       neighbour: list[int],
                       index1: int,
                       index2: int,
                       current_solution: list[int],
                       current_kemeny_score: int) -> int:
        """
        Efficiently calculate a kemeny score for a neighbour given an existing
        solution and corresponding kemeny score.

        This method takes less time than calc_kemeny_score() so should be used
        whenever possible.
        """
        if index1 == 0:
            index1 += 1
        remove_score: int = \
            self._calc_kemeny_score(current_solution[index1-1:index2+1])

        add_score: int = self._calc_kemeny_score(neighbour[index1-1:index2+1])

        new_kemeny_score: int = current_kemeny_score - remove_score + add_score

        return new_kemeny_score

    def _gen_two_change_neighbour(self,
                                  current_solution: list[int],
                                  current_kemeny: int) -> tuple[list[int],
                                                                int]:
        """
        Generate a 2 change neighbour by doing the following method:
        1. Pick two random non-adjacent edges to remove e.g.
           A-B-C-D-E -> A   B-C-D   E
        2. Reverse the interior chain A   D-C-B   E
        3. Reattach the edges in the chain A-D-C-B-E
        4. Calculate the kemeny score for the new solution
        """
        index1, index2 = self._gen_random_indexes()

        slice1: list[int] = current_solution[0:index1]
        slice2: list[int] = current_solution[index1:index2]
        slice3: list[int] = current_solution[index2:]

        reverse_slice2: list[int] = slice2[::-1]

        neighbour: list[int] = slice1 + reverse_slice2 + slice3

        # Perform feasability check if needed in this case not

        # Calculate kemeny_score for new neighbour
        new_kemeny_score: int = self._cost_function(neighbour,
                                                    index1,
                                                    index2,
                                                    current_solution,
                                                    current_kemeny)

        return neighbour, new_kemeny_score

    def _simulated_annealing(self, temp: float) -> list[int]:
        """
        Perform simulated annealing to find the lowest Kemeny score for a
        possible Kemeny ranking.
        """
        iter_with_no_improve: int = 0
        current_solution: list[int] = self._gen_initial_soluion()
        # Calculate the kemeny score of the initial solution
        current_kemeny_score: int = self._calc_kemeny_score(current_solution)
        # Outer loop where we check for stopping condition
        while iter_with_no_improve < self.num_non_improve:
            # Inner loop where we run for the temperature length
            for _ in range(0, self.temperature_length):
                # Generate a new 2-change neighbour (x'd)
                neighbour, new_kemeny_score = \
                    self._gen_two_change_neighbour(current_solution,
                                                   current_kemeny_score)
                cost_difference: int = new_kemeny_score - current_kemeny_score
                # Determine if the neighbour is a better solution
                if cost_difference <= 0:
                    # Accept neighbour as new best solution
                    current_solution = neighbour
                    current_kemeny_score = new_kemeny_score
                    iter_with_no_improve = 0
                else:
                    # Increment interations with no improvement for stopping
                    # condition
                    iter_with_no_improve += 1
                    # Randomly decide to accept neighbour
                    random_number: float = random.random()
                    if random_number < (math.e ** (-cost_difference/temp)):
                        # Accept neighbour as worse solution
                        current_solution = neighbour
                        current_kemeny_score = new_kemeny_score
            # Cool the temperature
            temp = self._cooling_ratio(temp)
        return current_solution

    def _print_solution(self,
                        best_solution: list[int],
                        best_kemeny_score: int):
        """
        Format the final solution and print in table form
        """

        max_padding: int = len(str(self.data_length))
        for driver_id in best_solution:
            profile: list[str] = self.drivers[driver_id-1].split(",")
            padding = " " * (max_padding - len(profile[0]))
            print(padding + profile[0] + " | " + profile[1])

        print("Best kemeny score:")
        print(best_kemeny_score)

    def run_simulations(self):
        """
        Controls the workflow when running the simulation.
        I chose to run multiple simulations simultaniously to increase the
        accuracy when working with random values.
        """
        threads: list = []
        results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.run_number) as executor:
            for _ in range(0, self.run_number):
                threads.append(executor.submit(self._simulated_annealing,
                                               self.initial_temperature))
            for future in concurrent.futures.as_completed(threads):
                solution = future.result()
                kemeny_score = self._calc_kemeny_score(solution)
                results[kemeny_score] = solution

        best_solution = results[min(results.keys())]
        best_kemeny_score = min(results.keys())

        self._print_solution(best_solution, best_kemeny_score)


if __name__ == "__main__":
    # Performance timer for benchmarking
    start: float = time.perf_counter()
    print("Processing...")

    sim_ann = SimulatedAnnealing()
    sim_ann.run_simulations()

    finish: float = time.perf_counter()
    print("Completed time in milliseconds:")
    print((finish - start) * 1000)
