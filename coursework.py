"""
Code for CM3109 Coursework by:
Matthew Battagel
C1946077

NOTE: You will need to pip install matplotlib graph feature
"""


import sys
import time
import math
import random
import concurrent.futures


class SimulatedAnnealing():
    """
    Stuff
    """

    def __init__(self):
        """
        Stuff
        """

        # Collect data for graph
        self.log: list[int] = []

        # Set SA Parameters
        self.INITIAL_TEMPERATURE: float = 10000
        self.TEMPERATURE_LENGTH: int = 1000
        self.TEMPERATURE_MULTIPLIER: float = 0.8
        self.NUM_NON_IMPROVE: int = 40
        self.RUN_NUMBER = 10

    def _collect_user_input(self):
        # Get file path containing data
        if len(sys.argv) < 2:
            print("Please enter the file path and try again")
            sys.exit()
        file_path: str = sys.argv[1]

        # Read data from file
        raw_data: list[str] = self._read_data(file_path)
        self.LENGTH, self.IDS, self.META, self.MATCHES = self._chunk_data(raw_data)
        self.SCORE_MATRIX: list[list[int]] = self._gen_score_matrix()

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
        while count <= self.LENGTH:
            gen_init_list.append(count)
            count += 1
        return gen_init_list

    def _cooling_ratio(self, temp: float) -> float:
        """
        Cooling ratio for reducing temperature over iterations
        """
        return self.TEMPERATURE_MULTIPLIER * temp

    def _gen_score_matrix(self) -> list[list[int]]:
        """
        Use the matches from the supplied file to create a matrix of all scores
        with size LENGTH * LENGTH

        Non-symmetrical as the score is only counted for the winner
        """
        # Generate empty matrix with size LENGHT * LENGTH
        score_matrix: list[list[int]] = [[0 for _ in range(self.LENGTH)]
                                         for _ in range(self.LENGTH)]
        # For each match add the weighting to matrix
        match: str
        for match in self.MATCHES:
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

    def debug_print_matrix(self, matrix: list[list[int]]):
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
        random_index1 = random.randint(0, self.LENGTH)
        random_index2 = random.randint(0, self.LENGTH)
        index_difference: int = abs(random_index1 - random_index2)
        if index_difference <= 1:
            # If the indices are unacceptable try again...
            smallest_index, largest_index = self._gen_random_indexes()
        else:
            # We DO want to hit the indexes of 0 and 46 as we want to hit
            # slices with no edges to ensure that the first and large person
            # in ranking is
            # changed.
            # [A-B-C-D]
            # Indexes of 0 and 46 and perform 2-change neighbour
            # [], [A-B-C-D], []
            # = [D-C-B-A]
            # Ranking is reversed which allows 1st and last to be changed
            smallest_index: int = min([random_index1, random_index2])
            largest_index: int = max([random_index1, random_index2])

        return smallest_index, largest_index

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
                disagreement: int = self.SCORE_MATRIX[driver2-1][driver1-1]
                if disagreement < 0:
                    print("Error - Comparing value with self.")
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

    def _gen_two_change_neighbour(self, current_solution: list[int],
                                 current_kemeny: int) -> tuple[list[int], int]:
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

    def _simulated_annealing(self,
                            current_solution: list[int],
                            temp: float) -> list[int]:
        """
        Perform simulated annealing to find the lowest Kemeny score for a
        possible Kemeny ranking.
        """
        print(current_solution)
        iter_with_no_improve: int = 0
        # Calculate the kemeny score of the initial solution
        current_kemeny_score = self._calc_kemeny_score(current_solution)
        # Outer loop where we check for stopping condition
        while iter_with_no_improve < self.NUM_NON_IMPROVE:
            # Inner loop where we run for the temperature length
            for _ in range(0, self.TEMPERATURE_LENGTH):
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
                    self.log.append(new_kemeny_score)
                    iter_with_no_improve = 0
                else:
                    # Increment interations with no improvement for stopping
                    # condition
                    iter_with_no_improve += 1
                    self.log.append(current_kemeny_score)
                    # Randomly decide to accept neighbour
                    random_number: float = random.random()
                    if random_number < (math.e ** (-cost_difference/temp)):
                        # Accept neighbour as worse solution
                        current_solution = neighbour
                        current_kemeny_score = new_kemeny_score
            # Cool the temperature
            temp = self._cooling_ratio(temp)
        return current_solution

    def print_solution(self, solution):
        """
        Format the final solution and print in table form
        """

        max_padding: int = len(str(self.LENGTH))
        for driver_id in solution:
            profile: list[str] = self.IDS[driver_id-1].split(",")
            padding = " " * (max_padding - len(profile[0]))
            print(padding + profile[0] + " | " + profile[1])

    def run_simulation(self) -> tuple[int, list[int]]:
        """
        Controls the workflow when running the simulation
        """
        self._collect_user_input()

        initial_solution = self._gen_initial_soluion()

        final_solution = self._simulated_annealing(initial_solution, self.INITIAL_TEMPERATURE)
        final_kemeny_score = self._calc_kemeny_score(final_solution)

        return final_kemeny_score, final_solution



if __name__ == "__main__":
        # Performance timer for benchmarking
        start: float = time.perf_counter()
        print("Processing...")

        RUN_NUMBER: int = 10

        sim_ann = SimulatedAnnealing()
        final_kemeny_score, final_solution = sim_ann.run_simulation():

        finish: float = time.perf_counter()
        print("Completed time in milliseconds:")
        print((finish - start) * 1000)

        # You can run multiple SA algorithms at a time
        # Change the constant RUN_NUMBER to increase iterations
        # for _ in range(0, RUN_NUMBER):
        #     final_solution: list[int] = simulated_annealing(current_solution,
        #                                                     INITIAL_TEMPERATURE)
        #     if keep_best_ranking:
        #         current_solution = final_solution
        #     final_kemeny_score = calc_kemeny_score(final_solution)
        #     results[final_kemeny_score] = final_solution

