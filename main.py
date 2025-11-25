from sys import argv

from simulator import RaceTrack, Simulator, plt

if __name__ == "__main__":
    assert(len(argv) == 2)
    racetrack = RaceTrack(argv[1])
    simulator = Simulator(racetrack, None)
    simulator.start()
    plt.show()
