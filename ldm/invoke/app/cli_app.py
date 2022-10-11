# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import argparse
import shlex
import os
from typing import Literal, Union, get_args, get_origin, get_type_hints
from pydantic import BaseModel
from pydantic.fields import Field
from .services.image_storage import DiskImageStorage
from .services.session_manager import DiskSessionManager
from .services.invocation_queue import MemoryInvocationQueue
from .invocations.baseinvocation import BaseInvocation
from .services.invocation_session import InvocationFieldLink
from .services.invocation_services import InvocationServices
from .services.invoker import Invoker, InvokerServices
from .invocations import *
from ..args import Args
from ...generate import Generate
from .services.events import EventServiceBase


class Command(BaseModel):
    invocation: Union[BaseInvocation.get_invocations()] = Field(discriminator="type")


class InvalidArgs(Exception):
    pass


def get_invocation_parser() -> argparse.ArgumentParser:

    # Create invocation parser
    parser = argparse.ArgumentParser()
    def exit(*args, **kwargs):
        raise InvalidArgs
    parser.exit = exit

    subparsers = parser.add_subparsers(dest='type')
    invocation_parsers = dict()

    invocations = BaseInvocation.get_all_subclasses()
    for invocation in invocations:
        hints = get_type_hints(invocation)
        cmd_name = get_args(hints['type'])[0]
        command_parser = subparsers.add_parser(cmd_name, help=invocation.__doc__)
        invocation_parsers[cmd_name] = command_parser


        fields = invocation.__fields__
        for name, field in fields.items():
            if name in ['id', 'type']:
                continue
            
            if get_origin(field.type_) == Literal:
                allowed_values = get_args(field.type_)
                allowed_types = set()
                for val in allowed_values:
                    allowed_types.add(type(val))
                allowed_types_list = list(allowed_types)
                field_type = allowed_types_list[0] if len(allowed_types) == 1 else Union[allowed_types_list]

                command_parser.add_argument(
                    f"--{name}",
                    dest=name,
                    type=field_type,
                    default=field.default,
                    choices = allowed_values,                    
                    help=field.field_info.description
                )
            else:
                command_parser.add_argument(
                    f"--{name}",
                    dest=name,
                    type=field.type_,
                    default=field.default,   
                    help=field.field_info.description
                )

    return parser


def invoke_cli():
    args = Args()
    config = args.parse_args()

    # TODO: this should move either inside ApiDependencies, or to their own invocations (or...their own services?)
    # Loading Face Restoration and ESRGAN Modules
    try:
        gfpgan, codeformer, esrgan = None, None, None
        if config.restore or config.esrgan:
            from ldm.invoke.restoration import Restoration
            restoration = Restoration()
            if config.restore:
                gfpgan, codeformer = restoration.load_face_restore_models(config.gfpgan_dir, config.gfpgan_model_path)
            else:
                print('>> Face restoration disabled')
            if config.esrgan:
                esrgan = restoration.load_esrgan(config.esrgan_bg_tile)
            else:
                print('>> Upscaling disabled')
        else:
            print('>> Face restoration and upscaling disabled')
    except (ModuleNotFoundError, ImportError):
        print('>> You may need to install the ESRGAN and/or GFPGAN modules')

    # Create Generate
    generate = Generate(
        model          = config.model,
        sampler_name   = config.sampler_name,
        embedding_path = config.embedding_path,
        full_precision = config.full_precision,
        gfpgan         = gfpgan,
        codeformer     = codeformer,
        esrgan         = esrgan
    )
    generate.free_gpu_mem = config.free_gpu_mem

    # NOTE: load model on first use, uncomment to load at startup
    # TODO: Make this a config option?
    #generate.load_model()

    events = EventServiceBase()

    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../outputs'))

    services = InvocationServices(
        generate = generate,
        events = events,
        images          = DiskImageStorage(output_folder)
    )

    invoker_services = InvokerServices(
        queue           = MemoryInvocationQueue(),
        session_manager = DiskSessionManager(output_folder)
    )

    invoker = Invoker(services, invoker_services)
    session = invoker.create_session()
    
    parser = get_invocation_parser()

    # Uncomment to print out previous sessions at startup
    # print(invoker_services.session_manager.list())

    while True:
        try:
            cmd_input = input("> ")
        except KeyboardInterrupt:
            # Ctrl-c exits
            break

        if cmd_input in ['exit','q']:
            break;

        if cmd_input in ['--help','help','h','?']:
            parser.print_help()
            continue

        try:
            # Split the command for piping
            cmds = cmd_input.split('|')
            start_id = len(session.history)
            current_id = start_id
            new_invocations = list()
            for cmd in cmds:
                # Parse args to create invocation
                args = vars(parser.parse_args(shlex.split(cmd.strip())))
                args['id'] = current_id
                command = Command(invocation = args)

                # Pipe previous command output (if there was a previous command)
                links = []
                if len(session.history) > 0 or current_id != start_id:
                    from_id = -1 if current_id == start_id else str(current_id - 1)
                    links.append(InvocationFieldLink(
                        from_node_id=from_id,
                        from_field = "*",
                        to_field = "*"))

                new_invocations.append((command.invocation, links))

                current_id = current_id + 1

            # Command line was parsed successfully
            # Add the invocations to the session
            for invocation in new_invocations:
                session.add_invocation(invocation[0], invocation[1])

            # Execute all available invocations
            invoker.invoke(session, invoke_all = True)
            session.wait_for_all()

        except InvalidArgs:
            print('Invalid command, use "help" to list commands')
            continue

        except SystemExit:
            continue
    
    invoker.stop()


if __name__ == "__main__":
    invoke_cli()
