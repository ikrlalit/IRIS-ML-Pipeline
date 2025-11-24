wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.headers["accept"] = "application/json"

wrk.body = [[
{
  "data": [5.1, 3.5, 1.4, 0.2]
}
]]
