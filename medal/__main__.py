from . import cmdline


if __name__ == "__main__":
    cmdline.main(cmdline.build_arg_parser().parse_args())
