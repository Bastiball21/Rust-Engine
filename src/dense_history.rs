use std::sync::OnceLock;

/// Dense move indexing for history tables.
///
/// Maps (from, to) squares to a compact ID space that excludes "impossible" moves.
/// A move is considered valid if it is a knight jump OR a queen move (rook-like or bishop-like).
///
/// Invalid moves map to INVALID (u16::MAX).
pub struct MoveIndexer;

const INVALID: u16 = u16::MAX;

static LOOKUP: OnceLock<([[u16; 64]; 64], usize)> = OnceLock::new();

impl MoveIndexer {
    #[inline(always)]
    fn init() -> ([[u16; 64]; 64], usize) {
        let mut table = [[INVALID; 64]; 64];
        let mut next_id: u16 = 0;

        for from in 0u8..64u8 {
            let from_file = (from & 7) as i8;
            let from_rank = (from >> 3) as i8;

            for to in 0u8..64u8 {
                if from == to {
                    continue;
                }

                let to_file = (to & 7) as i8;
                let to_rank = (to >> 3) as i8;

                let df = (to_file - from_file).abs();
                let dr = (to_rank - from_rank).abs();

                // Knight jump
                let is_knight = (df == 1 && dr == 2) || (df == 2 && dr == 1);

                // Queen move: same file, same rank, or diagonal
                let is_queen = (df == 0 && dr != 0) || (dr == 0 && df != 0) || (df == dr && df != 0);

                if is_knight || is_queen {
                    table[from as usize][to as usize] = next_id;
                    next_id = next_id.wrapping_add(1);
                }
            }
        }

        (table, next_id as usize)
    }

    /// Returns the number of dense move IDs (valid pseudo-legal moves for *any* piece).
    #[inline(always)]
    pub fn max_dense_moves() -> usize {
        LOOKUP.get_or_init(Self::init).1
    }

    /// Get the dense index for (from, to). Returns None if invalid.
    #[inline(always)]
    pub fn get_index(from: u8, to: u8) -> Option<usize> {
        let (table, _) = LOOKUP.get_or_init(Self::init);
        let v = table[from as usize][to as usize];
        if v == INVALID {
            None
        } else {
            Some(v as usize)
        }
    }
}
