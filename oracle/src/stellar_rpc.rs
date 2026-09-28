use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Clone)]
pub enum RpcError {
    NetworkError(String),
    HttpError {
        status: u16,
        body: String,
    },
    JsonError(String),
    RpcFault {
        code: i64,
        message: String,
    },
    BalanceBelowMinimum {
        balance_stroops: i64,
        balance_xlm: f64,
        min_xlm: f64,
    },
}

impl Eq for RpcError {}

/// Network errors and server-side/rate-limit HTTP statuses are transient and
/// worth retrying; parse errors, RPC faults, and balance-precondition
/// failures are not going to change on a bare retry, so they fail fast.
impl crate::retry::Retryable for RpcError {
    fn is_retryable(&self) -> bool {
        match self {
            RpcError::NetworkError(_) => true,
            RpcError::HttpError { status, .. } => *status >= 500 || *status == 429,
            RpcError::JsonError(_) => false,
            RpcError::RpcFault { .. } => false,
            RpcError::BalanceBelowMinimum { .. } => false,
        }
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RpcError::NetworkError(msg) => write!(f, "network error: {msg}"),
            RpcError::HttpError { status, body } => {
                if body.is_empty() {
                    write!(f, "HTTP {status}")
                } else {
                    write!(f, "HTTP {status}: {body}")
                }
            }
            RpcError::JsonError(msg) => write!(f, "JSON parse error: {msg}"),
            RpcError::RpcFault { code, message } => {
                write!(f, "RPC fault {code}: {message}")
            }
            RpcError::BalanceBelowMinimum {
                balance_xlm,
                min_xlm,
                ..
            } => {
                write!(
                    f,
                    "balance {balance_xlm} XLM is below minimum {min_xlm} XLM"
                )
            }
        }
    }
}

impl std::error::Error for RpcError {}

// ── JSON-RPC wire types ──────────────────────────────────────────────────────

#[derive(Serialize)]
pub(crate) struct JsonRpcRequest<'a, P: Serialize = serde_json::Value> {
    pub(crate) jsonrpc: &'a str,
    pub(crate) id: u32,
    pub(crate) method: &'a str,
    pub(crate) params: P,
}

#[derive(Deserialize)]
pub(crate) struct JsonRpcResponse<T> {
    pub(crate) result: Option<T>,
    pub(crate) error: Option<JsonRpcFault>,
}

#[derive(Deserialize)]
pub(crate) struct JsonRpcFault {
    pub(crate) code: i64,
    pub(crate) message: String,
}

// ── getLatestLedger ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
struct GetLatestLedgerResult {
    sequence: u32,
    #[allow(dead_code)]
    id: String,
    #[serde(rename = "protocolVersion")]
    #[allow(dead_code)]
    protocol_version: serde_json::Value,
}

/// Parse the raw JSON body returned by a `getLatestLedger` RPC call.
///
/// Kept separate from the HTTP layer so it can be unit-tested without
/// mocking the network.
pub fn parse_latest_ledger_response(body: &str) -> Result<u32, RpcError> {
    let resp: JsonRpcResponse<GetLatestLedgerResult> =
        serde_json::from_str(body).map_err(|e| RpcError::JsonError(e.to_string()))?;

    if let Some(fault) = resp.error {
        return Err(RpcError::RpcFault {
            code: fault.code,
            message: fault.message,
        });
    }

    resp.result
        .ok_or_else(|| RpcError::JsonError("missing 'result' field".to_string()))
        .map(|r| r.sequence)
}

/// Call `getLatestLedger` on the Stellar RPC endpoint and return the current
/// ledger sequence number.
///
/// **Caching note:** call this once per price-update cycle and pass the
/// returned value to any downstream function that needs `ledger_seq`.  This
/// avoids redundant round-trips within a single scheduled invocation.
pub async fn get_latest_ledger_sequence(rpc_url: &str) -> Result<u32, RpcError> {
    let payload = serde_json::to_string(&JsonRpcRequest {
        jsonrpc: "2.0",
        id: 1,
        method: "getLatestLedger",
        params: serde_json::Value::Object(serde_json::Map::new()),
    })
    .map_err(|e| RpcError::JsonError(e.to_string()))?;

    let body = rpc_post(rpc_url, payload).await?;
    parse_latest_ledger_response(&body)
}

/// Low-level helper: POST a JSON string to the RPC URL, return the response body.
pub(crate) async fn rpc_post(rpc_url: &str, payload: String) -> Result<String, RpcError> {
    let response = crate::http::client()
        .post(rpc_url)
        .header("Content-Type", "application/json")
        .body(payload)
        .send()
        .await
        .map_err(|e| RpcError::NetworkError(e.to_string()))?;

    let status = response.status().as_u16();
    let body = response
        .text()
        .await
        .map_err(|e| RpcError::NetworkError(e.to_string()))?;

    if status != 200 {
        return Err(RpcError::HttpError {
            status,
            body: crate::http::truncate_error_body(&body),
        });
    }

    Ok(body)
}

// ── Account balance (Horizon REST) ──────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct HorizonBalanceEntry {
    asset_type: String,
    balance: String,
}

#[derive(Debug, Deserialize)]
struct HorizonAccountResponse {
    balances: Vec<HorizonBalanceEntry>,
}

/// Parse the JSON body returned by `GET /accounts/{id}` on a Horizon server.
///
/// Returns the native (XLM) balance in stroops (1 XLM = 10_000_000 stroops).
pub fn parse_account_balance_response(body: &str) -> Result<i64, RpcError> {
    let resp: HorizonAccountResponse =
        serde_json::from_str(body).map_err(|e| RpcError::JsonError(e.to_string()))?;

    let native = resp
        .balances
        .iter()
        .find(|b| b.asset_type == "native")
        .ok_or_else(|| RpcError::JsonError("no native balance entry".to_string()))?;

    parse_xlm_balance_to_stroops(&native.balance)
}

/// Stroops in one XLM: 1 XLM = 10^7 stroops.
const STROOPS_PER_XLM: i64 = 10_000_000;
/// Fractional digits Horizon emits for XLM balances ("100.0000000"), which is
/// exactly stroop precision.
const HORIZON_XLM_DECIMALS: usize = 7;

/// Convert a Horizon XLM balance string into an exact stroop count.
///
/// Horizon renders XLM with exactly [`HORIZON_XLM_DECIMALS`] fractional digits,
/// which is precisely stroop precision, so the decimal string can be converted
/// with pure integer arithmetic and no rounding. The previous `f64` round-trip
/// introduced IEEE-754 error for balances that are not exactly representable in
/// binary floating point, yielding a stroop value off by one or more from the
/// true on-chain balance (#736, #899).
fn parse_xlm_balance_to_stroops(balance: &str) -> Result<i64, RpcError> {
    let raw = balance;
    let unparseable = || RpcError::JsonError(format!("unparseable balance: {raw}"));

    let balance = balance.trim();
    let (whole, fraction) = match balance.split_once('.') {
        Some((whole, fraction)) => (whole, fraction),
        None => (balance, ""),
    };

    if whole.is_empty()
        || !whole.bytes().all(|b| b.is_ascii_digit())
        || !fraction.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(unparseable());
    }

    let whole: i64 = whole.parse().map_err(|_| unparseable())?;

    // Horizon emits exactly 7 fractional digits, but pad anything shorter with
    // zeros and drop anything longer (sub-stroop precision) so the value stays
    // exact for well-formed input.
    let fraction = fraction.as_bytes();
    let mut fraction_stroops = 0i64;
    for position in 0..HORIZON_XLM_DECIMALS {
        let digit = fraction.get(position).copied().unwrap_or(b'0');
        fraction_stroops = fraction_stroops * 10 + i64::from(digit - b'0');
    }

    whole
        .checked_mul(STROOPS_PER_XLM)
        .and_then(|stroops| stroops.checked_add(fraction_stroops))
        .ok_or_else(unparseable)
}

/// Fetch the XLM balance for `account_id` from the Horizon server at
/// `horizon_url`.  Returns the balance in stroops.
pub async fn get_account_balance_stroops(
    horizon_url: &str,
    account_id: &str,
) -> Result<i64, RpcError> {
    let url = format!("{horizon_url}/accounts/{account_id}");

    let response = crate::http::client()
        .get(&url)
        .send()
        .await
        .map_err(|e| RpcError::NetworkError(e.to_string()))?;

    let status = response.status().as_u16();
    let body = response
        .text()
        .await
        .map_err(|e| RpcError::NetworkError(e.to_string()))?;

    if status != 200 {
        return Err(RpcError::HttpError {
            status,
            body: crate::http::truncate_error_body(&body),
        });
    }

    parse_account_balance_response(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that a valid `getLatestLedger` RPC response is parsed correctly
    /// and the sequence number is extracted. Closes #409.
    #[test]
    fn parse_valid_latest_ledger_response() {
        let body = r#"{
            "jsonrpc":"2.0","id":1,
            "result":{"id":"abc123","sequence":12345,"protocolVersion":22}
        }"#;
        assert_eq!(parse_latest_ledger_response(body).unwrap(), 12345u32);
    }

    /// Verifies that an RPC fault in the `getLatestLedger` response is
    /// propagated as `RpcError::RpcFault` with the correct code and message.
    /// Closes #410.
    #[test]
    fn parse_rpc_fault_response() {
        let body = r#"{
            "jsonrpc":"2.0","id":1,
            "error":{"code":-32000,"message":"start height out of range"}
        }"#;
        let err = parse_latest_ledger_response(body).unwrap_err();
        assert_eq!(
            err,
            RpcError::RpcFault {
                code: -32000,
                message: "start height out of range".to_string(),
            }
        );
    }

    #[test]
    fn parse_malformed_json_returns_error() {
        let err = parse_latest_ledger_response("not json").unwrap_err();
        assert!(matches!(err, RpcError::JsonError(_)));
    }

    #[test]
    fn parse_missing_result_field() {
        let body = r#"{"jsonrpc":"2.0","id":1}"#;
        let err = parse_latest_ledger_response(body).unwrap_err();
        assert!(matches!(err, RpcError::JsonError(_)));
    }

    // ── account balance ──────────────────────────────────────────────────────

    #[test]
    fn parse_account_balance_native_xlm() {
        let body = r#"{
            "id": "GABC",
            "balances": [
                {"asset_type":"credit_alphanum4","asset_code":"USDC","balance":"50.0000000"},
                {"asset_type":"native","balance":"100.5000000"}
            ]
        }"#;
        let stroops = parse_account_balance_response(body).unwrap();
        assert_eq!(stroops, 1_005_000_000);
    }

    #[test]
    fn parse_account_balance_low_balance() {
        let body = r#"{
            "id": "GABC",
            "balances": [{"asset_type":"native","balance":"0.5000000"}]
        }"#;
        let stroops = parse_account_balance_response(body).unwrap();
        assert_eq!(stroops, 5_000_000);
    }

    #[test]
    fn parse_account_balance_no_native_entry() {
        let body = r#"{
            "id": "GABC",
            "balances": [{"asset_type":"credit_alphanum4","asset_code":"USDC","balance":"10.0"}]
        }"#;
        let err = parse_account_balance_response(body).unwrap_err();
        assert!(matches!(err, RpcError::JsonError(_)));
    }

    #[test]
    fn parse_account_balance_malformed_json() {
        let err = parse_account_balance_response("not json").unwrap_err();
        assert!(matches!(err, RpcError::JsonError(_)));
    }

    /// #736/#899 - Horizon's balance string has exactly stroop precision, so it
    /// must not be round-tripped through `f64`. `319202939.9467375` is not
    /// representable in binary64; the old implementation returned
    /// `3192029399467376`, one stroop more than the true balance.
    #[test]
    fn parse_account_balance_is_exact_for_binary64_unrepresentable_value() {
        let body = r#"{
            "id": "GABC",
            "balances": [{"asset_type":"native","balance":"319202939.9467375"}]
        }"#;
        assert_eq!(
            parse_account_balance_response(body).unwrap(),
            3_192_029_399_467_375
        );
    }

    /// The same defect in the other rounding direction: this balance has no
    /// exact binary64 representation and previously parsed one stroop low.
    #[test]
    fn parse_account_balance_is_exact_when_f64_rounds_down() {
        let body = r#"{
            "id": "GABC",
            "balances": [{"asset_type":"native","balance":"883983974.2838425"}]
        }"#;
        assert_eq!(
            parse_account_balance_response(body).unwrap(),
            8_839_839_742_838_425
        );
    }

    #[test]
    fn parse_account_balance_pads_short_fraction_to_stroops() {
        let body = r#"{"id":"GABC","balances":[{"asset_type":"native","balance":"0.5"}]}"#;
        assert_eq!(parse_account_balance_response(body).unwrap(), 5_000_000);
    }

    #[test]
    fn parse_account_balance_accepts_whole_xlm() {
        let body = r#"{"id":"GABC","balances":[{"asset_type":"native","balance":"100"}]}"#;
        assert_eq!(parse_account_balance_response(body).unwrap(), 1_000_000_000);
    }

    #[test]
    fn parse_account_balance_truncates_sub_stroop_precision() {
        let body = r#"{"id":"GABC","balances":[{"asset_type":"native","balance":"1.00000009"}]}"#;
        assert_eq!(parse_account_balance_response(body).unwrap(), 10_000_000);
    }

    #[test]
    fn parse_account_balance_accepts_i64_max_stroops() {
        // i64::MAX = 9223372036854775807 stroops = 922337203685.4775807 XLM.
        let body = r#"{
            "id": "GABC",
            "balances": [{"asset_type":"native","balance":"922337203685.4775807"}]
        }"#;
        assert_eq!(parse_account_balance_response(body).unwrap(), i64::MAX);
    }

    #[test]
    fn parse_account_balance_rejects_malformed_values() {
        for balance in ["", "abc", "1.2.3", "-1.0000000", "1,5", "1e7", ".5"] {
            let body = format!(
                r#"{{"id":"GABC","balances":[{{"asset_type":"native","balance":"{balance}"}}]}}"#
            );
            let err = parse_account_balance_response(&body).unwrap_err();
            assert!(
                matches!(err, RpcError::JsonError(_)),
                "expected JsonError for balance {balance:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn parse_account_balance_rejects_stroop_overflow() {
        let body = r#"{
            "id": "GABC",
            "balances": [{"asset_type":"native","balance":"922337203686.0000000"}]
        }"#;
        let err = parse_account_balance_response(body).unwrap_err();
        assert!(matches!(err, RpcError::JsonError(_)));
    }

    // ── get_account_balance_stroops — HTTP-level tests (#404) ────────────────

    #[tokio::test]
    async fn get_account_balance_stroops_200_returns_stroops() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let body = r#"{
            "id": "GABC",
            "balances": [
                {"asset_type":"native","balance":"50.0000000"}
            ]
        }"#;

        Mock::given(method("GET"))
            .and(path("/accounts/GABC"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "application/json"))
            .mount(&server)
            .await;

        let result = get_account_balance_stroops(&server.uri(), "GABC").await;
        assert_eq!(result.unwrap(), 500_000_000); // 50 XLM in stroops
    }

    #[tokio::test]
    async fn get_account_balance_stroops_404_returns_http_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/accounts/GNOT_FOUND"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let err = get_account_balance_stroops(&server.uri(), "GNOT_FOUND")
            .await
            .unwrap_err();
        assert!(matches!(err, RpcError::HttpError { status: 404, .. }));
    }

    #[tokio::test]
    async fn get_account_balance_stroops_500_returns_http_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/accounts/GABC"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let err = get_account_balance_stroops(&server.uri(), "GABC")
            .await
            .unwrap_err();
        assert!(matches!(err, RpcError::HttpError { status: 500, .. }));
    }
}
