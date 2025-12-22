from utils.variant import Variant


trait Processor(Copyable & Movable):
    """A trait for operations that can process a quantum state."""

    fn process(self, mut state: Int) raises:
        ...


struct Processor1(Processor):
    var name: String

    fn __init__(out self):
        self.name = "Processor1"

    fn process(self, mut state: Int) raises:
        print("{} adds 1 to: ".format(self.name), state)
        state += 1


struct Processor2(Processor):
    var name: String

    fn __init__(out self):
        self.name = "Processor2"

    fn process(self, mut state: Int) raises:
        print("{} adds 2 to: ".format(self.name), state)
        state += 2


alias OneOf = Variant[Int, Bool]

alias Transformation = Variant[Processor1, Processor2, OneOf]


struct TransformationGroup(Processor):
    var transformations: List[Transformation]

    fn __init__(out self, lst: List[Transformation]):
        self.transformations = List[Transformation]()
        for i in range(len(lst)):
            self.transformations.append(lst[i])

    fn process(self, mut state: Int) raises:
        for processor in self.transformations:
            if processor.isa[Processor1]():
                processor[Processor1].process(state)
            elif processor.isa[Processor2]():
                processor[Processor2].process(state)


fn main() raises:
    print("Hello Mojo")
    processors: List[Transformation] = [Processor1(), Processor2(), OneOf(True)]
    var state = 0
    print("Initial state: ", state)
    TransformationGroup(processors).process(state)
    print("Final state: ", state)
