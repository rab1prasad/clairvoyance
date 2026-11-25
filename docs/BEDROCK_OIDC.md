# AWS Bedrock over OIDC — no static AWS keys on the pod

The pods run on GKE; Bedrock lives in AWS. Instead of long-lived
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, a pod can present its
Kubernetes service-account token (an OIDC JWT signed by the GKE cluster) to
AWS STS and assume an IAM role — `AssumeRoleWithWebIdentity`. The
credentials are short-lived and refresh themselves; nothing secret is stored.

## What the app does

Nothing OIDC-specific. A Bedrock block that resolves to **no API key**
(`BedrockAccount(api_key=None)` — no `credential_id`, no `api_key_name`)
runs on the pod's default AWS credential chain
(`app/ai/voice/llm/bedrock.py`). botocore's chain already includes the
web-identity provider: when `AWS_ROLE_ARN` and `AWS_WEB_IDENTITY_TOKEN_FILE`
are set, it exchanges the token file for role credentials and re-exchanges
before they expire, re-reading the file each time.

All default-chain Bedrock services in a process share **one** session
(`_default_chain_session`). botocore caches credentials per session, and
pipecat builds a fresh session per service — i.e. per call — so without
sharing, every call's first LLM turn would wait on an STS round trip. Shared,
the exchange happens once per pod (on its first Bedrock call) and then only
on refresh. Services with a Bedrock API key keep their own private session,
so keys never mix.

## Setup

Infra only; run by whoever owns the GKE cluster and the AWS account.

**1. The cluster's issuer.**

```bash
kubectl get --raw /.well-known/openid-configuration | jq -r .issuer
# https://container.googleapis.com/v1/projects/<PROJECT>/locations/<LOC>/clusters/<CLUSTER>
```

GKE serves the discovery document and JWKS publicly at that URL, so AWS can
verify the tokens without network access to the cluster.

**2. Register the issuer with AWS IAM** (once per cluster).

```bash
aws iam create-open-id-connect-provider \
  --url https://container.googleapis.com/v1/projects/<PROJECT>/locations/<LOC>/clusters/<CLUSTER> \
  --client-id-list sts.amazonaws.com
```

**3. A role only this service account can assume.** `<ISSUER>` is the URL
without `https://`.

Trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "arn:aws:iam::<ACCT>:oidc-provider/<ISSUER>" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": { "StringEquals": {
      "<ISSUER>:aud": "sts.amazonaws.com",
      "<ISSUER>:sub": "system:serviceaccount:<NAMESPACE>:<KSA_NAME>"
    }}
  }]
}
```

Permissions: `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`
(Converse and ConverseStream authorize as these). A cross-region inference
profile (`apac.*`, `global.*`) needs the inference-profile ARN **and** the
foundation-model ARN in every region the profile routes to.

**4. The pod: a projected token and two env vars.**

```yaml
spec:
  serviceAccountName: <KSA_NAME>
  containers:
    - name: clairvoyance
      env:
        - { name: AWS_ROLE_ARN, value: "arn:aws:iam::<ACCT>:role/<ROLE>" }
        - { name: AWS_WEB_IDENTITY_TOKEN_FILE, value: /var/run/secrets/aws/token }
      volumeMounts:
        - { name: aws-token, mountPath: /var/run/secrets/aws, readOnly: true }
  volumes:
    - name: aws-token
      projected:
        sources:
          - serviceAccountToken:
              audience: sts.amazonaws.com
              expirationSeconds: 3600
              path: token
```

The kubelet rotates the token file; botocore re-reads it on each refresh.

## What silently defeats it

- **Static keys win.** pipecat passes `os.getenv("AWS_ACCESS_KEY_ID")` /
  `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` to the client explicitly, so
  if they are in the pod's environment the role is never assumed. Remove
  them. (Their only other reader, `app/services/aws/utils.py`, serves AWS KMS,
  used only when `CLOUD_ENVIRONMENT=AWS`.)
- **A Bedrock API key wins.** A block with a `credential_id` row carrying
  `api_key`, or an `api_key_name`, authenticates with that bearer token and
  skips the chain. The build log names the path:
  `Building Bedrock LLM service … auth=default_chain` (OIDC) vs `auth=api_key`.
- **STS must be reachable.** The exchange goes to `sts.<region>.amazonaws.com`
  over the same egress as Bedrock itself.

## Verify

From inside a pod:

```bash
kubectl exec -it <pod> -- python -c \
  "import boto3; print(boto3.client('sts').get_caller_identity()['Arn'])"
# arn:aws:sts::<ACCT>:assumed-role/<ROLE>/botocore-session-…
```

An IAM user ARN instead of `assumed-role/…` means static keys are still in
the environment.

## Rolling back

Put the static keys back in the pod's environment; they take precedence over
the web-identity variables, which can stay in place. No code change.
