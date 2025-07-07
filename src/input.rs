/**************************************************************************/
/*  This file is part of POPCON.                                          */
/*                                                                        */
/*  Copyright (C) 2025                                                    */
/*    CEA (Commissariat à l'énergie atomique et aux énergies              */
/*         alternatives)                                                  */
/*                                                                        */
/*  you can redistribute it and/or modify it under the terms of the GNU   */
/*  Lesser General Public License as published by the Free Software       */
/*  Foundation, version 2.1.                                              */
/*                                                                        */
/*  It is distributed in the hope that it will be useful,                 */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of        */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         */
/*  GNU Lesser General Public License for more details.                   */
/*                                                                        */
/*  See the GNU Lesser General Public License version 2.1                 */
/*  for more details (enclosed in the file licenses/LGPLv2.1).            */
/*                                                                        */
/**************************************************************************/

//! Utils for reasonably easy streaming parser

use std::io::Read;

/// A buffer than can be used to feed a streaming parser which eats data at most `chunk_size` at
/// once.
pub struct ReadToParser<R: Read> {
    /// data is in the range index..end
    buffer: Vec<u8>,
    /// data is in the range index..end
    index: usize,
    /// data is in the range index..end
    end: usize,
    /// underlying byte source
    read: R,
    /// Number of \n that were forgotten as the buffer is reused
    lines: usize,
}

impl<R: Read> ReadToParser<R> {
    /// Creates a `ReadToParser` which can feed a nom parser by chunks of `chunk_size`.
    pub fn new(read: R, chunk_size: usize) -> Self {
        Self {
            buffer: vec![0; chunk_size],
            index: 0,
            end: 0,
            read,
            lines: 0,
        }
    }

    /// length of actual data
    pub fn available(&self) -> usize {
        self.end - self.index
    }

    /// attempts to read one more time for the underlying reader
    /// and refill the buffer
    /// Returns true if at least one more byte is added to the buffer.
    fn refill(&mut self) -> std::io::Result<bool> {
        let mut at_least_a_byte_read = false;
        if self.index != 0 {
            // we are going to erase lines
            self.lines += self.buffer[0..self.index]
                .iter()
                .filter(|&&x| x == b'\n')
                .count();
            self.buffer.copy_within(self.index..self.end, 0);
            self.end -= self.index;
            self.index = 0;
        }
        loop {
            match self.read.read(&mut self.buffer[self.end..]) {
                Ok(0) => {
                    return Ok(at_least_a_byte_read);
                }
                Ok(n) => {
                    self.end += n;
                    at_least_a_byte_read = true;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
            if self.end == self.buffer.len() {
                return Ok(at_least_a_byte_read);
            }
        }
    }

    /// Retrieve the line/column for index in buffer
    fn position(&self, index: usize) -> (usize, usize) {
        let mut line = 1 + self.lines;
        let mut column = 0;
        for &c in self.buffer[0..index].iter() {
            column += 1;
            if c == b'\n' {
                column = 0;
                line += 1;
            }
        }
        (line, column)
    }

    /// run the parser, and advance the file descriptor by how much the parser has consumed. Fails
    /// is the parser needs more than `chunk_size` at a time, or if the parser reaches eEOF and still can consume characters, then return None.
    /// Fails if if the parser fails, or if reading
    /// fails. Once an error
    /// is returned, no guarantee is done on the state of `self` the reader.
    fn parse<'a, O>(
        &'a mut self,
        parser: impl for<'b> Fn(&'b [u8]) -> nom::IResult<&'b [u8], O>,
    ) -> anyhow::Result<Option<O>> {
        if self.available() == 0 {
            self.refill()?;
        }
        let content = &self.buffer[self.index..self.end];
        match parser(content) {
            Ok((remaining, output)) => {
                self.index = self.end - remaining.len();
                Ok(Some(output))
            }
            Err(e) => match e {
                nom::Err::Incomplete(_) => Ok(None),
                nom::Err::Error(ee) | nom::Err::Failure(ee) => {
                    let remainder =
                        String::from_utf8_lossy(&ee.input[0..(std::cmp::min(20, ee.input.len()))]);
                    let (line, column) = self.position(self.end - ee.input.len());
                    anyhow::bail!(
                        "Could not parse {:?} (line {}, column {}) as {:?}",
                        remainder,
                        line,
                        column,
                        ee.code
                    )
                }
            },
        }
    }

    /// run the parser, and advance the file descriptor by how much the parser has consumed. Fails
    /// is the parser needs more than `chunk_size` at a time, or if the parser fails, or if reading
    /// fails.  Note that if the parser reaches EOF and could still consume characters (for example
    /// a parser that matches `a*` in the string "aaa") then an error is returned.  Once an error
    /// is returned, no guarantee is done on the state of `self` the reader.
    pub fn parse_and_advance<'a, O>(
        &'a mut self,
        parser: impl for<'b> Fn(&'b [u8]) -> nom::IResult<&'b [u8], O>,
    ) -> anyhow::Result<O> {
        loop {
            // this is not an infinite loop because refill will end up returning false.
            match self.parse(&parser) {
                Ok(Some(x)) => return Ok(x),
                Err(e) => return Err(e),
                Ok(None) => {}
            };
            let changed = self.refill()?;
            if !changed {
                anyhow::bail!("no enough bytes to read")
            }
        }
    }

    /// runs the parser repeatedly and discards its output until EOF is reached. Fails
    /// is the parser needs more than `chunk_size` at a time, or if it does not reach EOF, or if the parser fails, or if reading
    /// fails.
    pub fn parse_and_exhaust<'a, O>(
        mut self,
        parser: impl for<'b> Fn(&'b [u8]) -> nom::IResult<&'b [u8], O>,
    ) -> anyhow::Result<()> {
        loop {
            // this is not an infinite loop because refill will end up returning false.
            match self.parse(&parser) {
                Ok(Some(_)) => {}
                Err(e) => return Err(e),
                Ok(None) => {
                    let changed = self.refill()?;
                    if !changed {
                        if self.available() == 0 {
                            // EOF
                            return Ok(());
                        } else {
                            anyhow::bail!("no enough bytes to read")
                        }
                    }
                }
            };
        }
    }
}

#[test]
fn test_simple() {
    use nom::bytes::streaming::take_while1;
    use nom::character::{is_digit, is_space};
    use nom::IResult;
    use nom::Parser;

    fn number(s: &[u8]) -> IResult<&[u8], usize> {
        nom::combinator::map_res(take_while1(is_digit), |x: &[u8]| {
            let s: &str = std::str::from_utf8(x).unwrap();
            s.parse()
        })(s)
    }
    fn whitespace(s: &[u8]) -> IResult<&[u8], ()> {
        nom::combinator::map(take_while1(is_space), |_| ())(s)
    }
    fn empty_line(s: &[u8]) -> IResult<&[u8], ()> {
        nom::combinator::map(nom::bytes::streaming::tag(b"\n"), |_| ())(s)
    }
    fn whitespace_and_number(s: &[u8]) -> IResult<&[u8], ((), usize)> {
        whitespace.and(number).parse(s)
    }

    let mut r = ReadToParser::new(b"12345 67890\n" as &[u8], 8);
    assert_eq!(r.parse_and_advance(number).unwrap(), 12345);
    assert_eq!(r.parse_and_advance(whitespace).unwrap(), ());
    assert_eq!(r.parse_and_advance(number).unwrap(), 67890);
    assert_eq!(r.parse_and_exhaust(empty_line).unwrap(), ());

    let mut r = ReadToParser::new(b"12345 67890\ndon't drop some data" as &[u8], 8);
    assert_eq!(r.parse_and_advance(number).unwrap(), 12345);
    assert_eq!(r.parse_and_advance(whitespace).unwrap(), ());
    assert_eq!(r.parse_and_advance(number).unwrap(), 67890);
    assert!(r.parse_and_exhaust(empty_line).is_err());

    let mut r = ReadToParser::new(b"123 abc" as &[u8], 4);
    assert_eq!(r.parse_and_advance(number).unwrap(), 123);
    // this must refill to detect the error
    assert!(r.parse_and_advance(whitespace_and_number).is_err());
}
