package com.Fintech.OnlineBanking.auth;

import java.io.IOException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import com.Fintech.OnlineBanking.config.JwtService;
import com.Fintech.OnlineBanking.user.User;
import com.fasterxml.jackson.core.exc.StreamWriteException;
import com.fasterxml.jackson.databind.DatabindException;
import io.jsonwebtoken.ExpiredJwtException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@RestController
@RequestMapping("/user")
public class AuthenticationController {
	@Autowired
	private AuthenticationService service;
	@Autowired
	private JwtService jwtService;

	@PostMapping("/authenticate")
	public ResponseEntity<?> authenticationRequest(@RequestBody AuthenticationRequest request) {
		return service.authenticate(request);

	}

	@PostMapping("/refresh-token")
	public void refreshToken(HttpServletRequest request, HttpServletResponse response)
			throws StreamWriteException, DatabindException, IOException {
		service.refreshToken(request, response);

	}

	@GetMapping("/validate")
	public ResponseEntity<?> validateToken1(@RequestParam String token, @AuthenticationPrincipal User user)
			throws ExpiredJwtException {

		try {
			Boolean isTokenValid = jwtService.isTokenValid(token, user);
			Boolean isTokenNotRevoked = service.isRefreshTokenNotRevoked(token);
			return ResponseEntity.ok(isTokenValid && isTokenNotRevoked);
		}

		catch (ExpiredJwtException ex) {
			return ResponseEntity.ok(false);
		}

	}

}
