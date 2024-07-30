import os
from llama_index.core.tools import FunctionTool

def read_file(filename):
    # Open the file in read mode ('r')
    with open(filename, 'r') as file:
        # Read the contents of the file
        contents = file.read()

    # return the contents of the file
    return(contents)

def write_file(filename, content):
    # Open the file in write mode ('w')
    with open(filename, 'w') as file:
        # Write the content to the file
        file.write(content)

# define function tool
# function tool would help to define any custom code into Tool
read_file_tool = FunctionTool.from_defaults(
    fn = read_file,
    name = 'read_file_tool',
    description= 'This would help to read the content fo the given file'
)

write_file_tool = FunctionTool.from_defaults(
    fn = write_file,
    name = 'write_file_tool',
    description= 'This would help to write the content in the file'
)
