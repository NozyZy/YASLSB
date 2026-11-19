#!/usr/bin/env python3
"""
Stéganographie - Yet Another Stupid LSB

Embed / extract data from RGB images using the Least Significant Bit (LSB).
Supported traversal modes:
    - horizontal
    - vertical
    - diagonal
    - spiral
    - zigzag

The original behavior of the three provided scripts is preserved as much as
possible, while consolidating duplicated code into a single implementation.
"""

import argparse
import os.path

from PIL import Image


DEFAULT_OUTPUT = "embedded.png"
DIRECTIONS = ["horizontal", "vertical", "diagonal", "spiral", "zigzag"]
CHANNELS = {"R": 0, "G": 1, "B": 2}


def log(message, verbosity=0, level=0):
    """Print message when the configured verbosity is at least level."""
    if verbosity >= level:
        print(message)


# ---------------------------------------------------------------------------
# Bit conversion helpers
# ---------------------------------------------------------------------------

def convert_bytes_to_bits(data):
    return ''.join(format(i, '08b') for i in data)


def convert_str_to_bits(text):
    return ''.join(format(ord(i), '08b') for i in text)


def convert_bits_to_str(bits):
    return ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(bits)] * 8))


def change_int_value_bit(bits, value, position=0):
    if value not in '01':
        raise ValueError('Value needs to be 0 or 1.')
    if not 0 <= position <= 7:
        raise ValueError('Position needs to be between 0 and 7 included.')

    s = list(get_bits_from_int(bits))
    position = 7 - position
    s[position] = value
    return int(''.join(s), 2)


def get_bits_from_int(value):
    return f'{value:08b}'


def bitstring_to_bytes(s):
    s = s.ljust(len(s) + (8 - (len(s) % 8)), '0')
    return int(s, 2).to_bytes(len(s) // 8, byteorder='big')


# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------

def read_input_data(text=None, file_path=None, verbosity=0):
    """Read either literal text or file content."""
    if file_path:
        mode = 'r' if file_path.endswith('.txt') else 'rb'
        with open(file_path, mode) as f:
            data = f.read()
        log(f'[+] Reading input file: {file_path}', verbosity, 1)
    elif text is not None:
        data = text
        log('[+] Reading text input', verbosity, 1)
    else:
        data = ''

    if verbosity >= 2:
        print('[+] Input data:', data)

    return data


def data_to_bits(data, verbosity=0):
    log(f'[+] Converting input data to bits (type: {type(data).__name__})', verbosity, 1)

    if type(data) == str:
        to_embed = convert_str_to_bits(data)
    else:
        try:
            to_embed = convert_bytes_to_bits(data)
        except Exception as exc:
            raise TypeError(f'Cannot convert data type {type(data)}') from exc

    if verbosity >= 2:
        print('[+] Data bits:', to_embed)

    return to_embed


def sanitize_bits(data, sanitize_count=12):
    """Remove a trailing run of identical bits, matching the original scripts."""
    if len(data) > 8 * (sanitize_count + 1):
        tail = data[len(data) - 8 * sanitize_count:]
        if tail == '0' * 8 * sanitize_count or tail == '1' * 8 * sanitize_count:
            return data[:len(data) - 8 * sanitize_count]
    return data



# ---------------------------------------------------------------------------
# Horizontal / vertical / diagonal embedding and extraction (Scripts 1/2)
# ---------------------------------------------------------------------------

def normalize_channels(channels):
    """Return validated RGB channel indexes, preserving user order."""
    if not channels:
        return [0, 1, 2]

    normalized = []
    for channel in channels:
        key = channel.upper()
        if key not in CHANNELS:
            raise ValueError(f'Invalid color: {channel}. Use R, G or B.')
        index = CHANNELS[key]
        if index not in normalized:
            normalized.append(index)

    return normalized


def embed_standard(img, data, direction, channels=None):
    """Embed bits using horizontal, vertical or diagonal traversal."""
    channels = normalize_channels(channels)

    if direction == 'diagonal':
        for i in range(len(data)):
            pixel = list(img.getpixel((i, i)))
            channel = channels[i % len(channels)]
            pixel[channel] = change_int_value_bit(pixel[channel], data[i])
            img.putpixel((i, i), tuple(pixel))
        return img

    total = 0
    for i in range(img.height):
        if total >= len(data):
            break
        for j in range(img.width):
            if total >= len(data):
                break
            couple = (j, i) if direction == 'horizontal' else (i, j)
            pixel = list(img.getpixel(couple))
            channel = channels[total % len(channels)]
            pixel[channel] = change_int_value_bit(pixel[channel], data[total])
            img.putpixel(couple, tuple(pixel))
            total += 1

    return img


def extract_standard(img, direction, output_path=None, channels=None, verbosity=0):
    """Extract bits using horizontal, vertical or diagonal traversal."""
    channels = normalize_channels(channels)

    if direction == 'diagonal':
        data = ''
        for i in range(img.height):
            pixel = img.getpixel((i, i))
            channel = channels[i % len(channels)]
            data += get_bits_from_int(pixel[channel])[-1]
        return data

    data = ''
    total = 0
    sanitize_count = 12 if output_path else 2

    for i in range(img.height):
        if (total > 8 * (sanitize_count + 1) and
                (data[len(data) - 8 * sanitize_count:] == '0' * 8 * sanitize_count or
                 data[len(data) - 8 * sanitize_count:] == '1' * 8 * sanitize_count)):
            data = data[:len(data) - 8 * sanitize_count]
            break

        for j in range(img.width):
            couple = (j, i) if direction == 'horizontal' else (i, j)
            pixel = img.getpixel(couple)
            channel = channels[total % len(channels)]
            data += get_bits_from_int(pixel[channel])[-1]
            total += 1

            if (total > 8 * (sanitize_count + 1) and
                    (data[len(data) - 8 * sanitize_count:] == '0' * 8 * sanitize_count or
                     data[len(data) - 8 * sanitize_count:] == '1' * 8 * sanitize_count)):
                data = data[:len(data) - 8 * sanitize_count]
                break

    return data


# ---------------------------------------------------------------------------
# Zigzag / spiral helpers (Script 3)
# ---------------------------------------------------------------------------

def max_zigzag_pixels(width, height):
    depth = height // 2
    total = (width * depth) + (2 * depth) - 2
    return total


def max_spiral_pixels(width, height):
    """Maximum number of pixels traversed by the original spiral routine.

    This follows the same shrinking-layer logic as the provided draw/read
    functions, but computes the capacity without writing into an image.
    """
    # Simulate the exact segment-length rules used by embed_spiral/extract_spiral.
    x_length = width - 1
    y_length = height - 1
    dir_index = 0
    total = 0

    while x_length > 0 or y_length > 0:
        if dir_index % 2 == 0:
            length = x_length
        else:
            length = y_length

        if length <= 0:
            break

        total += length

        if dir_index >= 1:
            if dir_index % 2 == 0:
                x_length -= 2
            else:
                y_length -= 2

        dir_index += 1

    return total


def embed_spiral(image_path, data, channels=None):
    img = Image.open(image_path).convert("RGB")

    channels = normalize_channels(channels)

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    x, y = 0, 0
    width, height = img.size
    dir_index = 0
    data_size = len(data)

    max_size = max_spiral_pixels(width, height)
    assert data_size <= max_size, (
        f"Cannot embed {data_size} bits, max is {max_size} for this image."
    )

    x_length = width - 1
    y_length = height - 1
    i = 0

    while i < data_size:
        dx, dy = directions[dir_index % 4]
        length = x_length if dx != 0 else y_length

        channel = channels[dir_index % len(channels)]

        for _ in range(length):
            if i >= data_size:
                break
            pixel = list(img.getpixel((x, y)))
            pixel[channel] = change_int_value_bit(pixel[channel], data[i])
            img.putpixel((x, y), tuple(pixel))
            print(f"({x},{y})", end=", ")
            x += dx
            y += dy
            i += 1

        if dir_index >= 1:
            match dir_index % 2:
                case 0:
                    x_length -= 2
                case 1:
                    y_length -= 2

        dir_index += 1
        print("\n")
    return img


def embed_zigzag(image_path, data, channels=None):
    img = Image.open(image_path).convert("RGB")
    channels = normalize_channels(channels)

    width, height = img.size
    data_size = len(data)

    assert data_size <= max_zigzag_pixels(width, height), (
        f"Cannot embed {data_size} bits, "
        f"max is {max_zigzag_pixels(width, height)} for this image."
    )

    i = 0
    y = 0
    channel_index = 0
    direction = 1

    while i < data_size and y < height:

        # ---------------------------------------------------------------
        # Ligne de données
        # ---------------------------------------------------------------
        x_start = 0 if direction == 1 else width - 1
        x_end = width if direction == 1 else -1

        for x in range(x_start, x_end, direction):
            if i >= data_size:
                break

            pixel = list(img.getpixel((x, y)))

            channel = channels[channel_index % len(channels)]

            pixel[channel] = change_int_value_bit(pixel[channel], data[i])

            img.putpixel((x, y), tuple(pixel))
            i += 1

        if i >= data_size:
            break

        # ---------------------------------------------------------------
        # Ligne de transition : un seul pixel
        # ---------------------------------------------------------------
        if y + 1 >= height:
            break

        transition_x = width - 1 if direction == 1 else 0

        y += 1
        channel_index += 1

        pixel = list(img.getpixel((transition_x, y)))

        channel = channels[channel_index % len(channels)]

        pixel[channel] = change_int_value_bit(pixel[channel], data[i])

        img.putpixel((transition_x, y), tuple(pixel))
        i += 1

        # Ligne suivante : changement de sens
        direction *= -1

        y += 1
        channel_index += 1

    return img


def extract_spiral(image_path, channels=None):
    img = Image.open(image_path).convert("RGB")
    channels = normalize_channels(channels)

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    x, y = 0, 0
    width, height = img.size
    dir_index = 0
    done = False

    data = ""
    sanitize_count = 12
    max_size = max_spiral_pixels(width, height)

    x_length = width - 1
    y_length = height - 1
    length = 0
    i = 0

    while i < max_size and length > -1 and not done:
        dx, dy = directions[dir_index % 4]
        length = x_length if dx != 0 else y_length
        # Same rule as embedding: one channel for the whole segment.
        channel = channels[dir_index % len(channels)]

        for _ in range(length):
            if i >= max_size or (i > 8 * (sanitize_count + 1) and
                    (data[len(data) - 8 * sanitize_count:] == '0' * 8 * sanitize_count or
                     data[len(data) - 8 * sanitize_count:] == '1' * 8 * sanitize_count)):
                data = data[:len(data) - 8 * sanitize_count]
                done = True
                break

            pixel = img.getpixel((x, y))
            data += get_bits_from_int(pixel[channel])[-1]
            x += dx
            y += dy
            i += 1

        if dir_index >= 1:
            match dir_index % 2:
                case 0:
                    x_length -= 2
                case 1:
                    y_length -= 2

        dir_index += 1

    return data


def extract_zigzag(image_path, channels=None):
    img = Image.open(image_path).convert("RGB")
    channels = normalize_channels(channels)

    width, height = img.size

    data = ""
    y = 0
    channel_index = 0
    direction = 1

    sanitize_count = 12
    max_size = max_zigzag_pixels(width, height)

    while len(data) < max_size and y < height:

        # ---------------------------------------------------------------
        # Ligne de données
        # ---------------------------------------------------------------
        x_start = 0 if direction == 1 else width - 1
        x_end = width if direction == 1 else -1

        for x in range(x_start, x_end, direction):
            if len(data) >= max_size:
                break

            # Détection de la séquence de terminaison, comme dans
            # les autres modes d'extraction.
            if (
                len(data) > 8 * (sanitize_count + 1)
                and (
                    data[-8 * sanitize_count:] ==
                    '0' * 8 * sanitize_count
                    or
                    data[-8 * sanitize_count:] ==
                    '1' * 8 * sanitize_count
                )
            ):
                return data[:-8 * sanitize_count]

            pixel = img.getpixel((x, y))

            channel = channels[channel_index % len(channels)]

            data += get_bits_from_int(pixel[channel])[-1]

        # ---------------------------------------------------------------
        # Ligne de transition : un seul pixel
        # ---------------------------------------------------------------
        if y + 1 >= height:
            break

        transition_x = width - 1 if direction == 1 else 0

        y += 1
        channel_index += 1

        if len(data) >= max_size:
            break

        pixel = img.getpixel((transition_x, y))

        channel = channels[channel_index % len(channels)]

        data += get_bits_from_int(pixel[channel])[-1]

        # ---------------------------------------------------------------
        # Ligne suivante : changement de sens
        # ---------------------------------------------------------------
        direction *= -1

        y += 1
        channel_index += 1

    return data


# ---------------------------------------------------------------------------
# Generic dispatch
# ---------------------------------------------------------------------------

def embed_image(image_path, data, direction, channels=None):
    """Embed bits according to the selected traversal mode."""
    if direction in ('horizontal', 'vertical', 'diagonal'):
        img = Image.open(image_path).convert('RGB')
        return embed_standard(img, data, direction, channels=channels)

    if direction == 'spiral':
        return embed_spiral(image_path, data, channels=channels)

    if direction == 'zigzag':
        return embed_zigzag(image_path, data, channels=channels)

    raise ValueError(f'Unsupported direction: {direction}')


def extract_data(image_path, direction, output_path=None, channels=None, verbosity=0):
    """Extract bits according to the selected traversal mode."""
    if direction in ('horizontal', 'vertical', 'diagonal'):
        img = Image.open(image_path).convert('RGB')
        return extract_standard(
            img, direction, output_path=output_path, channels=channels, verbosity=verbosity
        )

    if direction == 'spiral':
        return extract_spiral(image_path, channels=channels)

    if direction == 'zigzag':
        return extract_zigzag(image_path, channels=channels)

    raise ValueError(f'Unsupported direction: {direction}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog='LSB embedder and extractor',
        description=(
            'Extracts and embeds data into PNG images using LSB, '
            'horizontally, vertically, diagonally, in spiral or zigzag.'
        )
    )

    parser.add_argument(
        'image',
        type=str,
        help='Image to embed/extract data'
    )
    parser.add_argument(
        '-t', '--text',
        type=str,
        default=None,
        required=False,
        help='Text to be embedded'
    )
    parser.add_argument(
        '-f', '--file',
        type=str,
        default=None,
        required=False,
        help='File to be embedded (.txt in plain text, other in binary)'
    )
    parser.add_argument(
        '-d', '--direction',
        type=str,
        default='horizontal',
        choices=DIRECTIONS,
        nargs='?',
        help='Direction / traversal used to embed or extract data'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=DEFAULT_OUTPUT,
        required=False,
        help='Output image or extracted data file'
    )
    parser.add_argument(
        '-c', '--color', '--colors',
        dest='colors',
        nargs='+',
        type=str.upper,
        choices=['R', 'G', 'B'],
        metavar='RGB',
        default=['R', 'G', 'B'],
        help=(
            'RGB color channel(s) used for LSB data. Multiple channels are '
            'cycled in the order given (default: R G B).'
        )
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help=(
            'Increase verbosity: -v=level 1, -vv=level 2, '
            '-vvv=level 3 (default: level 0).'
        )
    )

    return parser


def validate_args(parser, args):
    if not os.path.isfile(args.image):
        parser.error(f'Image {args.image} does not exists.')

    if args.file and not os.path.isfile(args.file):
        parser.error(f'File {args.file} does not exists.')

    if args.text is not None and args.file is not None:
        parser.error('Use either --text or --file, not both.')


def run(args):
    parser = build_parser()
    validate_args(parser, args)

    img = Image.open(args.image).convert('RGB')
    embedding = args.text is not None or args.file is not None

    log(f'[+] Mode: {"embedding" if embedding else "extraction"}', args.verbose, 0)
    log(f'[+] Image: {args.image}', args.verbose, 0)
    log(f'[+] Direction: {args.direction}', args.verbose, 0)
    log(f'[+] Channels: {" ".join(args.colors)}', args.verbose, 0)
    log(f'[+] Image size: {img.width}x{img.height}', args.verbose, 0)
    log(f'[+] Output: {args.output}', args.verbose, 0)

    if args.direction == 'diagonal' and img.height != img.width:
        log('[!] Diagonal extraction and embedding only works on square images.', args.verbose, 0)

    if embedding:
        data = read_input_data(args.text, args.file, args.verbose)
        to_embed = data_to_bits(data, args.verbose)

        if isinstance(data, (bytes, bytearray)):
            input_size = f'{len(data)} bytes'
        else:
            input_size = f'{len(data)} characters'
        log(f'[+] Input size: {input_size}', args.verbose, 0)
        log(f'[+] Payload size: {len(to_embed)} bits', args.verbose, 0)

        if args.direction == 'diagonal':
            size = img.width
            max_size = size // 8
            if len(to_embed) > size:
                parser.error(f'Your text not should exceed {size // 8} for that image in diagonal.')
        elif args.direction == 'spiral':
            max_size = max_spiral_pixels(img.width, img.height)
            log(f'[+] Spiral capacity: {max_size} bits', args.verbose, 1)
            if len(to_embed) > max_size:
                parser.error(f'Cannot embed {len(to_embed)} bits, max is {max_size} for this image.')
        elif args.direction == 'zigzag':
            max_size = max_zigzag_pixels(img.width, img.height)
            log(f'[+] Zigzag capacity: {max_size} bits', args.verbose, 1)
            if len(to_embed) > max_size:
                parser.error(f'Cannot embed {len(to_embed)} bits, max is {max_size} for this image.')
        elif len(to_embed) > img.width * img.height:
            parser.error('Not enough space to embed data.')

        log(f'[+] Embedding data fill: {len(to_embed) / max_size * 100:.2f}%')
        log('[+] Embedding payload into image', args.verbose, 1)
        embedded = embed_image(args.image, to_embed, args.direction, channels=args.colors)
        embedded.save(args.output, 'PNG')
        log(f'[+] Data embedded successfully to {args.output}', args.verbose, 0)
        return

    # Extraction mode.
    log('[+] Extracting data', args.verbose, 0)
    log('[+] Reading LSB payload from image', args.verbose, 1)
    data = extract_data(
        args.image, args.direction, output_path=args.output, channels=args.colors, verbosity=args.verbose
    )

    log(f'[+] Extracted payload size: {len(data)} bits', args.verbose, 0)
    if args.verbose >= 2:
        print('[+] Extracted data bits:', data)

    if not data:
        log('[-] No data was found.', args.verbose, 0)
    elif args.output != DEFAULT_OUTPUT:
        log('[+] Converting extracted bits to bytes', args.verbose, 1)
        extracted = bitstring_to_bytes(data)
        if args.verbose >= 2:
            print('[+] Extracted bytes:', extracted)
        with open(args.output, 'wb') as f:
            f.write(extracted)
        log(f'[+] Successfully extracted {len(extracted)} bytes to {args.output}', args.verbose, 0)
    else:
        log('[+] Converting extracted bits to text', args.verbose, 1)
        extracted_text = convert_bits_to_str(data)
        if args.verbose >= 2:
            print('[+] Extracted text:', extracted_text)
        else:
            print(f'[+] Extracted data : {extracted_text}')


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    run(args)


if __name__ == '__main__':
    main()
