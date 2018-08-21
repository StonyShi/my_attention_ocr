

try:
    from computer_text_generator import ComputerTextGenerator
    try:
        from handwritten_text_generator import HandwrittenTextGenerator
    except ImportError as e:
        print('Missing modules for handwritten text generation.')
    from background_generator import BackgroundGenerator
    from distorsion_generator import DistorsionGenerator
except Exception as e:
    import sys, os
    parentdir = os.path.dirname(os.path.abspath(__file__))
    # print("---------------")
    # print(parentdir)
    sys.path.insert(0, parentdir)
    from computer_text_generator import ComputerTextGenerator

    try:
        from handwritten_text_generator import HandwrittenTextGenerator
    except ImportError as e:
        print('Missing modules for handwritten text generation.')
    from background_generator import BackgroundGenerator
    from distorsion_generator import DistorsionGenerator

__all__ = [ComputerTextGenerator, BackgroundGenerator, DistorsionGenerator]
